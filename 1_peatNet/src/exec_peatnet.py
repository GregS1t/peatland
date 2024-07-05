import os, sys

import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox

from numba import jit

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import TensorDataset
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import the custom modules
from peatnet import *
from utils import *


# --------------------------------------------------------------------------
# Define the parameters
data_dir = "/home/gsainton/CALER/PEATMAP/1_NN_training/training_data"

learn_rate = 0.001          # Learning rate
hidden_size = 64            # Number of neurons in the hidden layers  
num_epochs = 5              # Number of epochs
HL = 2                      # Number of hidden layers
nb_file2merge = 2           # Number of files to merge
frac_samples = 15           # Fraction of the data to extract
normalize = True            # Normalize the data
verbose = True             # Verbose mode
carbon_estimation = True    # Estimate the carbon footprint
carbon_log_file = "carbon_footprint.log"
# --------------------------------------------------------------------------

if carbon_estimation:
    start = datetime.datetime.now()

# Select the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if verbose:
    print(f"Device: {device}")

# Load the dataset
list_files = get_files(data_dir)
if verbose: 
    print(f"\tThere are {len(list_files)} files in the directory")

# Select randomly a subset of the files
list_files_selected = np.random.choice(list_files, nb_file2merge, replace=False)
if verbose: 
    print(f"\tSelecting {nb_file2merge} files: {list_files_selected}")
    print("Stacking the data...")

X, y = stack_data(data_dir, list_files_selected, frac_samples=0.10, 
                  seed=42, verbose=verbose)

# Convert X and y to dataframes
X = pd.DataFrame(X)
y = pd.DataFrame(y)

X_fields = ['dist0005', 'dist0100', 'dist1000', 'hand0005',
    'hand0100', 'hand1000', 'slope',
    'elevation', 'wtd', 'landsat_1', 'landsat_2',
    'landsat_3', 'landsat_4', 'landsat_5', 'landsat_6',
    'landsat_7', 'NDVI']

X.columns = X_fields    
y_fields = ['peatland']
y.columns = y_fields

if normalize:
    print("Normalizing the data...")
    fields_to_transform = [ 'dist0005', 'dist0100', 'dist1000', 'hand0005',
    'hand0005', 'hand0100', 'hand1000', 'slope', 'wtd',
    'landsat_1', 'landsat_2', 'landsat_3', 'landsat_4',
    'landsat_7', 'NDVI']

    inplace = False
    if inplace:
        X = box_cox_transform(X, fields_to_transform)
    else:
        X_boxcox = X.copy()
        X_boxcox = box_cox_transform(X_boxcox, fields_to_transform)

    # Add a MinMaxScaler scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_boxcox)
    X_scaled = pd.DataFrame(X_scaled, columns=X_fields)


# Split the data into train, validation and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

if verbose:
    print("Data splitted into train, validation and test datasets")
    print("\t - Train dataset size:", len(X_train))
    print("\t - Validation dataset size:", len(X_val))
    print("\t - Test dataset size:", len(X_test))


# Define model parameters
input_size = list(X_train.shape)[1]
output_size = list(y_train.shape)[1] if len(y_train.shape) > 1 else 1

# Convert to tensors
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

if verbose:
    print("Creating DataLoaders...")

# Create DataLoader
train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
validate_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Free some memory
del X_train, X_val, X_test, y_train, y_val, y_test
del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

if verbose:
    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}")


# Create and train the model
# =============================================================================
model = PeatNet(input_size, hidden_size, output_size).to(device)

if verbose:
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(model)
    print("-."*50)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

if verbose:
    print("Start training the model...")
total_time = train_model(model, train_loader, validate_loader, 
                         criterion, optimizer, num_epochs=num_epochs, device=device)


train_model(model, train_loader, validate_loader, criterion, 
            optimizer, num_epochs=10, device='cuda', scheduler=scheduler)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Save the model
# =============================================================================
if verbose:
    print(f"Saving the model... to model_peatnet_{current_time}.ckpt")
torch.save(model.state_dict(), f"model_peatnet_{current_time}.ckpt")


# Test the model
# =============================================================================
if verbose:
    print("Testing the model...")
val_loss = 0.0
with torch.no_grad():
    for inputs, targets in tqdm(validate_loader, desc='Final Validation', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
avg_val_loss = val_loss / len(validate_loader)
if verbose:
    print(f'Final Validation Loss: {avg_val_loss:.4f}')


# Evaluate the model in term of carbon footprint
# =============================================================================
if carbon_estimation:
    end = datetime.datetime.now()
    total_training_time = (end-start).total_seconds()
    if verbose:
        print(f"Duration of the script: {end-start}")

    # Estimating the carbon footprint
    power_consumption_gpu = 300  # watts
    power_consumption_cpu = 150  # watts
    carbon_intensity = 0.32  # kg CO2 per kWh (estimation for France in 2023)

    if device.type == 'cpu':
        power_consumption = power_consumption_cpu
    else:
        power_consumption = power_consumption_gpu

    # Assuming the training was done on GPU
    total_energy_kwh = (power_consumption / 1000) * (total_training_time / 3600)  # convert time to hours and power to kW
    total_carbon_footprint = total_energy_kwh * carbon_intensity

    if verbose:
        print(f'Total energy consumed (estimated): {total_energy_kwh:.2f} kWh')
        print(f'Total carbon footprint (estimated): {total_carbon_footprint:.2f} kg CO2')
    
    # Create or append to the log file
    if os.path.exists(carbon_log_file):
        mode = 'a'
    else:
        mode = 'w'
    
    # Save the current time, the total energy consumed and the total carbon footprint and the name of the program 
    with open(carbon_log_file, mode) as f:
        f.write(f"{current_time}, {total_energy_kwh:.2f}, {total_carbon_footprint:.2f}, {sys.argv[0]}\n")

    if verbose:
        print(f"Carbon footprint saved in {carbon_log_file}")




