import os
import numpy as np
import torch
import h5py
from contextlib import closing


import os
import numpy as np
import torch
import h5py

from scipy.stats import boxcox


def make_subset_data(data_dir, datafile, frac_samples=0.3, seed=42, verbose=False):
    ''' 
    Load the data from a file and make a subset of the data
    INPUT: 
    ------
    @data_dir: str - Directory where the data is stored
    @datafile: str - Name of the file containing the data
    @frac_samples: float - Fraction of the data to extract
    @seed: int - Seed for the random number generator
    @verbose: bool - Print information about the data

    OUTPUT: 
    -------
    @X: numpy array - Input data
    @y: numpy array - Target data
    '''
    # Load data
    file_path = os.path.join(data_dir, datafile)
    with h5py.File(file_path, 'r') as f:
        input_data = np.array(f.get('input')).astype(np.float32).T
        target_data = np.array(f.get('target')).astype(np.float32).T

    nb_lines = input_data.shape[0]
    nb_lines2extract = int(nb_lines * frac_samples)
    if verbose:
        print(f"Filenames: {datafile}")
        print(f"Number of lines in the file: {nb_lines}")
        print(f"Number of lines to extract: {nb_lines2extract}")

    # Select a random subset of the data
    np.random.seed(seed)
    idx = np.random.choice(input_data.shape[0], nb_lines2extract, replace=False)
    X = input_data[idx]
    y = target_data[idx]
    if verbose:
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
    return X, y

def stack_data(data_dir, list_files, frac_samples=0.3, seed=42, verbose=False):
    '''
    Load the data from a list of files and stack the data
    INPUT:
    ------
    @data_dir: str - Directory where the data is stored
    @list_files: list - List of files containing the data
    @frac_samples: float - Fraction of the data to extract
    @seed: int - Seed for the random number generator
    @verbose: bool - Print information about the data

    OUTPUT:
    -------
    @X: numpy array - Input data
    @y: numpy array - Target data
    '''
    X, y = [], []
    for file in list_files:
        print(f"Processing file: {file} to extract {frac_samples * 100:.2f}% of the data")
        X_, y_ = make_subset_data(data_dir, file, frac_samples, seed, verbose=verbose)
        X.append(X_)
        y.append(y_)

    X = np.vstack(X)
    y = np.vstack(y)

    return X, y

def load_dataset_mat(data_dir, datafile):
    '''
    Load the data from a .mat file
    INPUT:
    ------
    @data_dir: str - Directory where the data is stored
    @datafile: str - Name of the file containing the data

    OUTPUT:
    -------
    @X: torch.tensor - Input data
    @y: torch.tensor - Target data
    '''

    file_path = os.path.join(data_dir, datafile)
    with h5py.File(file_path, 'r') as f:
        input_data = np.array(f.get('input')).astype(np.float32).T
        target_data = np.array(f.get('target')).astype(np.float32).T

    X = torch.from_numpy(input_data)
    y = torch.from_numpy(target_data)
        
    return X, y
        
def get_files(data_dir):
    '''
    Get the list of files in the data directory
    INPUT:
    ------
    @data_dir: str - Directory where the data is stored
    
    OUTPUT:
    -------
    @files: list - List of files in the data directory
    '''
    
    return [file for file in os.listdir(data_dir) if file.endswith(".mat") and file.startswith("trainingData")]


def box_cox_transform(X, fields_to_transform):
    '''
    Apply the Box-Cox transformation to the data
    INPUT:
    ------
    @X: pandas dataframe - Data to transform
    @fields_to_transform: list - List of fields to transform

    OUTPUT:
    -------
    @X: pandas dataframe - Transformed data
    '''
    
    for field in fields_to_transform:
        if any(X[field] <= 0):
            X[field] = X[field] - X[field].min() + 1
        X[field], _ = boxcox(X[field])
    return X