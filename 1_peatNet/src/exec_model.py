
import os, sys

import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as Data

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, random_split

import h5py

from pprint import pprint

from tqdm import tqdm

# Define a classe to define a neural network
class PeatNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PeatNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    


def make_subset_data(data_dir, datafile, frac_samples=0.3, seed=42, verbose=False):
    # Load data
    with h5py.File(os.path.join(data_dir, datafile), 'r') as f:
        input = f.get('input')
        arr_input = np.array(input)
        # convert to float64
        arr_input = arr_input.astype(np.float32)
        arr_input = arr_input.T
        
        latS = f.get('latS')
        arr_latS = np.array(latS)
        # convert to float32
        arr_latS = arr_latS.astype(np.float32)
        arr_latS = arr_latS.T

        lonS = f.get('lonS')
        arr_lonS = np.array(lonS)
        # convert to float32
        arr_lonS = arr_lonS.astype(np.float32)
        arr_lonS = arr_lonS.T
        
        # Concatenate the input data and the coordinates
        arr_input = np.concatenate((arr_input, arr_latS, arr_lonS), axis=1)

        target = f.get('target')
        arr_target = np.array(target)
        print(f"Shape of target: {arr_target.shape}")
        # convert to float32
        arr_target = arr_target.astype(np.float32)
        arr_target = arr_target.T

        # Convert to PyTorch tensors
        X = torch.from_numpy(arr_input)
        y = torch.from_numpy(arr_target)
    
    nb_lines = X.shape[0]
    nb_lines2extract = int(nb_lines * frac_samples)
    if verbose:
        print(f"Filenames: {datafile}  ")
        print(f"Number of lines in the file: {nb_lines}")
        print(f"Number of lines to extract: {nb_lines2extract}")

    # Select a random subset of the data
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], nb_lines2extract, replace=False)
    X = X[idx]
    y = y[idx]

    return X, y

def stack_data(data_dir, list_files, frac_samples=0.3, seed=42, 
               save2pd=False, output_dir="../outputs/", verbose=False):
    X = []
    y = []
    for file in list_files:
        X_, y_ = make_subset_data(data_dir, file, frac_samples, seed, verbose=verbose)
        X.append(X_)
        y.append(y_)

    
    X = torch.cat(X)
    y = torch.cat(y)

    if save2pd:
        # Save to pandas
        X_pd = X.numpy()
        y_pd = y.numpy()
        X_pd = pd.DataFrame(X_pd)

        #input fields
        X_fields = ['dist0005', 'dist0100', 'dist1000', 'hand0005',
          'hand0100', 'hand1000', 'slope',
          'elevation', 'wtd', 'landsat_1', 'landsat_2',
          'landsat_3', 'landsat_4', 'landsat_5', 'landsat_6',
          'landsat_7', 'NDVI', 'lat', 'lon']
        X_pd.columns = X_fields    

        #target fields
        y_pd = pd.DataFrame(y_pd)
        y_fields = ['peatland']
        y_pd.columns = y_fields

        # Create a timestamp
        date_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filenameX = os.path.join(output_dir, f"X_merge_{date_now}.csv")
        print(f"Ouput file: {filenameX}")
        filenamey = os.path.join(output_dir, f"y_merge_{date_now}.csv")
        X_pd.to_csv(filenameX)
        y_pd.to_csv(filenamey)
        if verbose:
            print(f"Data saved to {output_dir}")
            print(f"X shape: {X_pd.shape}")
            print(f"y shape: {y_pd.shape}")
        return X, y, filenameX, filenamey
    else: 
        return X, y, None, None


def load_dataset_mat(data_dir, datafile):
    with h5py.File(os.path.join(data_dir, datafile), 'r') as f:
        input = f.get('input')
        arr_input = np.array(input)
        # convert to float64
        arr_input = arr_input.astype(np.float32)
        arr_input = arr_input.T
        
        target = f.get('target')
        arr_target = np.array(target)
        # convert to float32
        arr_target = arr_target.astype(np.float32)
        arr_target = arr_target.T

        # Convert to PyTorch tensors
        X = torch.from_numpy(arr_input)
        del(arr_input)    
        y = torch.from_numpy(arr_target)
        del(arr_target)
        
    return X, y


def make_subset_data(data_dir, datafile, frac_samples=0.3, seed=42, verbose=False):
    # Load data
    with h5py.File(os.path.join(data_dir, datafile), 'r') as f:
        input = f.get('input')
        arr_input = np.array(input)
        # convert to float64
        arr_input = arr_input.astype(np.float32)
        arr_input = arr_input.T
        
        latS = f.get('latS')
        arr_latS = np.array(latS)
        # convert to float32
        arr_latS = arr_latS.astype(np.float32)
        arr_latS = arr_latS.T

        lonS = f.get('lonS')
        arr_lonS = np.array(lonS)
        # convert to float32
        arr_lonS = arr_lonS.astype(np.float32)
        arr_lonS = arr_lonS.T
        
        # Concatenate the input data and the coordinates
        arr_input = np.concatenate((arr_input, arr_latS, arr_lonS), axis=1)

        target = f.get('target')
        arr_target = np.array(target)
        print(f"Shape of target: {arr_target.shape}")
        # convert to float32
        arr_target = arr_target.astype(np.float32)
        arr_target = arr_target.T

        # Convert to PyTorch tensors
        X = torch.from_numpy(arr_input)
        y = torch.from_numpy(arr_target)
    
    nb_lines = X.shape[0]
    nb_lines2extract = int(nb_lines * frac_samples)
    if verbose:
        print(f"Filenames: {datafile}  ")
        print(f"Number of lines in the file: {nb_lines}")
        print(f"Number of lines to extract: {nb_lines2extract}")

    # Select a random subset of the data
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], nb_lines2extract, replace=False)
    X = X[idx]
    y = y[idx]

    return X, y


def load_dataset_pandas(datadir, filenameX, filenamey):
    X = pd.read_csv(filenameX)
    y = pd.read_csv(filenamey)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Drop the first column
    X = X.drop(columns=X.columns[0])
    y = y.drop(columns=y.columns[0])
    

    # Convert to PyTorch tensors
    X = torch.from_numpy(X.values)
    y = torch.from_numpy(y.values)

    # Transpose the tensors
    X = X.T
    y = y.T

    return X, y


def split_dataset_with_val(dataset, train_ratio=0.7, validate_ratio=0.15, 
                           test_ratio=0.15, random_seed=None):
    """
    Splits a PyTorch TensorDataset into train, validation, and test subsets.

    @dataset (TensorDataset): The dataset to split.
    @train_ratio (float): The ratio of the dataset to include in the training set.
    @validate_ratio (float): The ratio of the dataset to include in the validation set.
    @test_ratio (float): The ratio of the dataset to include in the test set.
    @random_seed (int): The random seed to use for splitting the dataset.

    Returns:
    - train_set (TensorDataset): The training set.
    - validate_set (TensorDataset): The validation set.
    - test_set (TensorDataset): The test set.

    Usage:
    train_set, validate_set, test_set = split_dataset_with_val(dataset, 
                                                              train_ratio=0.7, 
                                                              validate_ratio=0.15, 
                                                              test_ratio=0.15,
                                                              random_seed=42) 

    """

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    validate_size = int(total_size * validate_ratio)
    test_size = total_size - train_size - validate_size

    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Use random_split to split the dataset into train, validation, and test sets
    train_set, validate_set, test_set = random_split(
        dataset, [train_size, validate_size, test_size])

    return train_set, validate_set, test_set

# Define the training function  
def train_old(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    loss_array = np.zeros(num_epochs)
    epoch_array = np.arange(num_epochs)
    
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 2000 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        loss_array[epoch] = loss.item()
        epoch_array[epoch] = epoch

    return loss_array, epoch_array


def calculate_accuracy(outputs, targets):
    # Assuming binary classification with threshold at 0.5
    predicted = torch.round(outputs)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    epochs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
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
        
        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_progress = tqdm(val_dataloader, desc=f'Validation {epoch+1}/{num_epochs}', leave=False)
        with torch.no_grad():
            for inputs, targets in val_progress:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        epochs.append(epoch + 1)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    
    return epochs, train_losses, val_losses





def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    """
    Trains a neural network model using the specified training and validation data loaders.

    @model (nn.Module): The neural network model to train.
    @train_loader (DataLoader): The DataLoader for the training dataset.
    @val_loader (DataLoader): The DataLoader for the validation dataset.
    @criterion (nn.Module): The loss function to use.
    @optimizer (torch.optim.Optimizer): The optimizer to use.
    @num_epochs (int): The number of epochs to train the model.

    Returns:
    - epoch_array (np.ndarray): The array of epoch numbers.
    - train_loss_array (np.ndarray): The array of training loss values.
    - val_loss_array (np.ndarray): The array of validation loss values.
    - train_acc_array (np.ndarray): The array of training accuracy values.
    - val_acc_array (np.ndarray): The array of validation accuracy values.

    Usage:
    epoch_array, train_loss_array, val_loss_array, train_acc_array, val_acc_array = train(model, 
                                                                                          train_loader, 
                                                                                          val_loader, 
                                                                                          criterion, 
                                                                                          optimizer, 
                                                                                          num_epochs)
    """


    device = next(model.parameters()).device
    model.train()
    
    train_loss_array = []
    val_loss_array = []
    train_acc_array = []
    val_acc_array = []
    epoch_array = np.arange(num_epochs)

    # Loop over the epochs
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True) as t:
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Make predictions
                outputs = model(images)

                # Compute the loss and perform backpropagation
                loss = criterion(outputs, labels)
                loss.backward()

                # Update the model parameters
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                t.set_postfix({'train_loss': train_loss / (t.n + 1), 'train_acc': 100. * correct_train / total_train})

        train_loss_array.append(train_loss / len(train_loader))
        train_accuracy = 100. * correct_train / total_train
        train_acc_array.append(train_accuracy)
        epoch_array[epoch] = epoch
        
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=True) as t:
                for images, labels in t:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

                    t.set_postfix({'val_loss': val_loss / (t.n + 1), 'val_acc': 100. * correct_val / total_val})

        val_loss_array.append(val_loss / len(val_loader))
        val_accuracy = 100. * correct_val / total_val
        val_acc_array.append(val_accuracy)

    return epoch_array, train_loss_array, val_loss_array, train_acc_array, val_acc_array


def get_files(data_dir):
    files = []
    for file in os.listdir(data_dir):
        if file.endswith(".mat") and file.startswith("trainingData"):
            files.append(file)
    return files


# Load the dataset
if __name__ == "__main__":
    
    #Starttime of the script
    start = datetime.datetime.now()
    
    data_dir = "/home/gsainton/CALER/PEATMAP/1_NN_training/training_data"
    #datafile = "trainingData_n50w100.mat"

    learn_rate = 0.005
    hidden_size = 10
    num_epochs = 2

    # Select the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load the dataset
    # Matlab file containing the dataset
    print("Loading dataset...")
    list_files = get_files(data_dir)
    pprint(list_files)
    print(f"There are {len(list_files)} files in the directory")


    X, y, filenameX, filenamey = stack_data(data_dir, list_files[0:1], frac_samples=0.10, seed=42, 
                  save2pd=True, verbose=True)

    #X, y = load_dataset_mat(data_dir, datafile)

    #sys.exit("Stop here")


    torch_dataset = Data.TensorDataset(X, y)
    print("Splitting dataset...")
    train_set, validate_set, test_set = split_dataset_with_val(torch_dataset, 
                                                            train_ratio=0.8, 
                                                            validate_ratio=0.10, 
                                                            test_ratio=0.10,
                                                            random_seed=42)

    # Normalize train_set
    print("Normalizing train dataset...")
    mean = train_set.dataset.tensors[0].mean(dim=0)
    std = train_set.dataset.tensors[0].std(dim=0)

    train_set_input_normalized = (train_set.dataset.tensors[0] - mean) / std
    train_set = torch.utils.data.TensorDataset(train_set_input_normalized, train_set.dataset.tensors[1])

    # Check the sizes of the resulting datasets
    print("\t - Train dataset size:", len(train_set))
    print("\t - Validation dataset size:", len(validate_set))
    print("\t - Test dataset size:", len(test_set))

    # Create a DataLoader for each dataset
    train_loader = Data.DataLoader(train_set, batch_size=64, 
                                   shuffle=True, num_workers=8)
    
    validate_loader = Data.DataLoader(validate_set, batch_size=64,
                                        num_workers=8)
    
    test_loader = Data.DataLoader(test_set, batch_size=64,
                                        num_workers=8)
    
    # Define model parameters
    input_size = list(X.shape)[1]
    output_size = list(y.shape)[1]
    
    # Create the model
    model = PeatNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # Train the model
    print("Training the model...")
    #loss_array, epoch_array = train(model, train_loader, criterion, optimizer, num_epochs)
    #epoch_array, train_loss_array, val_loss_array, train_acc_array, val_acc_array = train(model, train_loader, validate_loader, 
    #                                                                        criterion, optimizer, num_epochs)
    
    epoch_array, train_loss_array, val_loss_array = train_model(model, train_loader, validate_loader, 
                                                                criterion, optimizer, num_epochs=num_epochs)


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save the model
    print(f"Saving the model... to model{current_time}.ckpt")
    torch.save(model.state_dict(), f"model{current_time}.ckpt")
    
    # Test the model
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(validate_loader, desc='Final Validation', leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(validate_loader)
    print(f'Final Validation Loss: {avg_val_loss:.4f}')

    end = datetime.datetime.now()
    total_training_time = (end-start).total_seconds()
    print(f"Duration of the script: {end-start}")



    # Plot the loss and the accuracy in two subplots
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(epoch_array, train_loss_array, label='Train', color='blue')
    axs[0].plot(epoch_array, val_loss_array, label='Validation', color='red', linestyle='dashed')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # axs[1].plot(epoch_array, train_acc_array, label='Train', color='blue')
    # axs[1].plot(epoch_array, val_acc_array, label='Validation', color='red', linestyle='dashed')
    # axs[1].set_title('Accuracy')
    # axs[1].set_xlabel('Epoch')
    # axs[1].set_ylabel('Accuracy')
    # axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    HL = 2

    plt.savefig(f"../outputs/pealand_model_{HL}HL_with_{hidden_size}neur_{num_epochs}epoch.png")


    # Estimating the carbon footprint
    power_consumption_gpu = 300  # watts
    power_consumption_cpu = 150  # watts
    carbon_intensity = 0.32  # kg CO2 per kWh (estimation for France in 2023)

    if device.type == 'cpu':
        power_consumption = power_consumption_cpu
    else:
        power_consumption = power_consumption_gpu

    # Get the comsumption of the GPU card   
    


    # Assuming the training was done on GPU
    total_energy_kwh = (power_consumption / 1000) * (total_training_time / 3600)  # convert time to hours and power to kW
    total_carbon_footprint = total_energy_kwh * carbon_intensity
    print(f'Total energy consumed (estimated): {total_energy_kwh:.2f} kWh')
    print(f'Total carbon footprint (estimated): {total_carbon_footprint:.2f} kg CO2')
