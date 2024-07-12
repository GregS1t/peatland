
import os

import time

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import mean_absolute_error

from tqdm import tqdm

# Define a class to define a neural network
class PeatNet(nn.Module):
    '''
    This class defines a neural network with 3 layers.
    The first layer has input_size neurons, the second layer has hidden_size neurons 
    and the third layer has output_size neurons.
    The activation function is ReLU.
    '''

    def __init__(self, input_size, output_size):
        super(PeatNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        #out = self.dropout(out)
        out = self.fc3(out)
        return out

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, 
                num_epochs=10, device='cpu', scheduler=None):
    
    model.to(device)
    train_losses = []
    val_losses = []
    train_mae_scores = []
    val_mae_scores = []

    writer = SummaryWriter(log_dir='../runs/mlp_experiment')
    
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        running_loss = 0.0
        train_mae = 0.0  # Initialize MAE accumulator
        train_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}',
                              leave=False, total=len(train_dataloader))
        for i, (inputs, targets) in enumerate(train_progress):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_progress.set_postfix({'Batch Loss': loss.item()})

            train_mae += mean_absolute_error(outputs.cpu().detach().numpy(), 
                                             targets.cpu().detach().numpy())
        
        avg_train_loss = running_loss / len(train_dataloader)
        avg_train_mae = train_mae / len(train_dataloader)  # Calculate average MAE
        
        # Log metrics for training to Tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)
        writer.add_scalar('MAE/Train', avg_train_mae, epoch + 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0  # Initialize MAE accumulator

        val_progress = tqdm(val_dataloader, desc=f'Validation {epoch+1}/{num_epochs}', 
                            leave=False, total=len(val_dataloader))
        with torch.no_grad():
            for inputs, targets in val_progress:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_mae += mean_absolute_error(outputs.cpu().detach().numpy(), 
                                               targets.cpu().detach().numpy())

        avg_val_loss = val_loss / len(val_dataloader)   
        avg_val_mae = val_mae / len(val_dataloader)  # Calculate average MAE
        
        # Log metrics for validation to Tensorboard
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)
        writer.add_scalar('MAE/Validation', avg_val_mae, epoch + 1)

        if scheduler:
            scheduler.step()  # Update the learning rate scheduler

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, Training MAE: {avg_train_mae:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation MAE: {avg_val_mae:.4f}, Epoch Time: {epoch_duration:.2f} seconds')
    
    total_time = time.time() - start_time

    writer.close()
    return {
        'total_time': total_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mae_scores': train_mae_scores,
        'val_mae_scores': val_mae_scores
    }

def save_model(model: nn.Module, model_dir: str, current_time: str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fullname_model = os.path.join(model_dir, f"model_peatnet_{current_time}.ckpt")
    torch.save(model.state_dict(), fullname_model)
    return fullname_model
