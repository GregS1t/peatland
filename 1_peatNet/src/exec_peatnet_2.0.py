#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Information about the this code and the author
# Author : Grégory Sainton
# Date : 2024-06-20
# Description : Train a neural network to predict peatland

import sys
import os
import datetime
import logging
import argparse
import subprocess

from scipy.stats import boxcox # For the normalization

from numba import jit          # For possible speedup

import torch                   # All the torch imports
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import TensorDataset
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.model_selection import train_test_split # For the split of the data

# Personnal imports
from peatnet import *
from utils import *
from libCarbonFootprint import *


# ------------------------------------------------------------------------------
# Parameters
#
# Define the parameters of the model and the training
# Some of them are overwritten by the command line arguments
# ------------------------------------------------------------------------------
learn_rate = 0.001          # Learning rate
num_epochs = 5              # Number of epochs
nb_file2merge = 5           # Number of files to merge
frac_samples = 0.10         # Fraction of the data to extract
normalize = True            # Normalize the data
verbose = True              # Verbose mode

model_dir = "../peatnet_models"

carbon_estimation = True    # Estimate the carbon footprint
carbon_log_file = "carbon_footprint.log"
training_log_file = "peatnet_training"

data_dir = "/home/gsainton/CALER/PEATMAP/1_NN_training/training_data" if os.uname().nodename == 'ares6' else "/data/gsainton/PEATLAND_DATA"
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(f"training_log_file_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

if verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

def get_gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        return result.stdout
    except Exception as e:
        logger.error(f"Failed to execute nvidia-smi: {e}")
        sys.exit(1)

def is_gpu_free(gpu_id):
    gpu_usage = get_gpu_usage()
    logger.info(f"GPU usage:\n{gpu_usage}")
    # Check for the presence of any process using the specified GPU
    if f' No running processes found' in gpu_usage.split('GPU')[gpu_id + 1]:
        return True
    return False

def setup_device(mydevice: str) -> torch.device:
    mydevice = str(mydevice)
    if not torch.cuda.is_available():
        logger.error("GPU is not available -> device = 'cpu'...")
        device = torch.device('cpu')
    else:
        num_gpus = torch.cuda.device_count()
        logger.info("GPU found...")
        logger.info(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"Device {i} name: {torch.cuda.get_device_name(i)}")

        valid_devices = [f'cuda:{i}' for i in range(num_gpus)]
        logger.info(f"Valid GPU references: {valid_devices}")

        if str(mydevice) in valid_devices:
            gpu_id = int(mydevice.split(':')[1])
            if is_gpu_free(gpu_id):
                logger.info(f"Using GPU - {mydevice}")
                device = torch.device(mydevice)
            else:
                logger.error(f"GPU {mydevice} may be currently in use...")
                device = torch.device(mydevice)
        else:
            logger.error(f"Invalid GPU reference: {str(mydevice)} -> Valid references: {valid_devices}. Exiting...")
            sys.exit(1)

    return device

if __name__ == '__main__':

    # Get argument from the command line
    parser = argparse.ArgumentParser(description='Train a neural network to predict peatland')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--nb_file2merge', type=int, default=2, help='Number of files to merge')
    parser.add_argument('--frac_samples', type=float, default=0.10, help='Fraction of the data to extract')
    parser.add_argument('--normalize', type=bool, default=True, help='Normalize the data')
    parser.add_argument('--gpu_ref', type=str, default='cuda:0', help='GPU reference')
    args = parser.parse_args()

    num_epochs = args.num_epochs
    nb_file2merge = args.nb_file2merge
    frac_samples = args.frac_samples
    normalize = args.normalize
    mydevice = torch.device(args.gpu_ref)

    if frac_samples > 1 or frac_samples < 0:
        raise ValueError("frac_samples must be between 0 and 1")

    # Example of command line:
    # python exec_peatnet_2.0.py --num_epochs 2 --nb_file2merge 2 --frac_samples 0.10 --normalize True

    carbon_log_dir = "/home/gsainton/CARBON_LOG" if os.uname().nodename == 'ares6' else "/obs/gsainton/PEATLAND_DATA"

    if carbon_estimation:
        start = datetime.datetime.now()
    if not os.path.exists(carbon_log_dir):

        os.makedirs(carbon_log_dir)

    device = setup_device(mydevice)

    #---------------------------------------------------------------------------
    #
    # Load and preprocess the data
    #
    #---------------------------------------------------------------------------

    peatmat_data_proc = PeatNetDataProc(data_dir=data_dir, frac_samples=frac_samples)

    peatmat_data_proc.set_list_rdn_files(nb_file2merge)
    sub_sampled_data = peatmat_data_proc.get_list_rdn_files()

    # Create output data file with the parameter used and the date

    filename = f"output_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_epoch{num_epochs}_frac{frac_samples}_f2merge{nb_file2merge}.npy"


    outputdatafile = os.path.join(data_dir, "..", "output" + filename)
    X, y = peatmat_data_proc.load_data(save=True, outputfile=outputdatafile)

    logging.info("Number of tiles to merge : {}".format(nb_file2merge))
    logging.info("Fraction of samples to extract : {}".format(frac_samples))

    if len(sub_sampled_data) == 0 or nb_file2merge == 0:
        logging.error("No data to process. Check the input parameters or the data directory. Exiting...")
        sys.exit(1)
    logging.info("Files used for training :")
    for f in sub_sampled_data:
        logging.info("- {}".format(f))

    X_fields = ['dist0005', 'dist0100', 'dist1000', 'hand0005',
        'hand0100', 'hand1000', 'slope',
        'elevation', 'wtd', 'landsat_1', 'landsat_2',
        'landsat_3', 'landsat_4', 'landsat_5', 'landsat_6',
        'landsat_7', 'NDVI']
    # Après reflexion, j'ai enlevé les deux colonnes latS et lonS qui de mon
    # point de vue ne doivent pas être utilisées pour la prédiction de la
    # présence de tourbière

    X.columns = X_fields
    y_fields = ['peatland']
    y.columns = y_fields

    if normalize:
        logger.info("Normalizing the data requested (BoxCox + MinMaxScaler)...")
        fields_to_transform = [ 'dist0005', 'dist0100', 'dist1000', 'hand0005',
        'hand0005', 'hand0100', 'hand1000', 'slope', 'wtd',
        'landsat_1', 'landsat_2', 'landsat_3', 'landsat_4',
        'landsat_7', 'NDVI']
        X = peatmat_data_proc.normalize_data(X, fields_to_transform)

    # Split the data into train, validation and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    test_size=0.5,
                                                    random_state=42)

    logger.info("Data splitted into train, validation and test datasets")
    logger.info(f"Train dataset size: {len(X_train)}")
    logger.info(f"Validation dataset size:sub_sampled_data =  {len(X_val)}")
    logger.info(f"Test dataset size: {len(X_test)}")

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

    # Create DataLoader
    train_loader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    validate_loader = Data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = Data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Free some memories
    del X_train, X_val, X_test, y_train, y_val, y_test
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor

    # --------------------------------------------------------------------------
    #
    # Define and train the model
    #
    # --------------------------------------------------------------------------

    model = PeatNet(input_size, output_size).to(device)

    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.debug(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate,
                                 weight_decay=1e-5)
    logging.info(f"Optimizer: {optimizer}")
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    logger.info("Training the model...")
    total_time = train_model(model, train_loader, validate_loader,
                            criterion, optimizer, num_epochs=num_epochs,
                            device=device)

    train_model(model, train_loader, validate_loader, criterion, optimizer,
                num_epochs=num_epochs, device=device, scheduler=scheduler)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    fullname_model = save_model(model, model_dir, current_time)
    logger.info(f"Model saved to {fullname_model}")


    # --------------------------------------------------------------------------
    #
    # Test the model after training
    #
    # --------------------------------------------------------------------------

    logger.info("Testing the model...")
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(validate_loader, desc='Final Validation',
                                    leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_mae += mean_absolute_error(outputs.cpu().detach().numpy(),
                                           targets.cpu().detach().numpy())

    avg_val_loss = val_loss / len(validate_loader)
    avg_val_mae = val_mae / len(validate_loader)
    logger.info(f"Validation loss: {avg_val_loss:.4f}")
    logger.info(f"Validation MAE: {avg_val_mae:.4f}")


    # --------------------------------------------------------------------------
    #
    # Naive carbon footprint estimation
    #
    # --------------------------------------------------------------------------


    if carbon_estimation:
        carbon_footprint_calculator = CarbonFootprintCalculator(device)
        carbon_logger = CarbonFootprintLogger(carbon_log_dir, carbon_log_file)
        end, total_energy_kwh, total_carbon_footprint = carbon_footprint_calculator.calculate(start)
        carbon_logger.log_carbon_footprint(end, total_energy_kwh,
                                           total_carbon_footprint)

    logger.info("End of the script")
