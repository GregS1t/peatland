import os
import re
import numpy as np
import pandas as pd
import torch
import h5py
from contextlib import closing

from scipy.stats import boxcox

from sklearn.preprocessing import MinMaxScaler

class PeatNetDataProc:
    def __init__(self, data_dir, frac_samples=0.3, seed=42):
        self.data_dir = data_dir
        self.frac_samples = min(1, frac_samples)
        self.seed = seed
        self.list_rdn_files = []
        self.list_all_files = self.get_files()
        self.nb_file2merge = 0

    def make_subset_data(self, datafile):
        file_path = os.path.join(self.data_dir, datafile)
        with h5py.File(file_path, 'r') as f:
            input_data = np.array(f.get('input')).astype(np.float32).T
            target_data = np.array(f.get('target')).astype(np.float32).T
            # lat_data = np.array(f.get('latS')).astype(np.float32).T
            # lon_data = np.array(f.get('lonS')).astype(np.float32).T
            # input_data = np.concatenate((input_data, lat_data, lon_data), axis=1)

        nb_lines = input_data.shape[0]
        nb_lines2extract = int(nb_lines * self.frac_samples)

        np.random.seed(self.seed)
        idx = np.random.choice(input_data.shape[0], nb_lines2extract, replace=False)
        X = input_data[idx]
        y = target_data[idx]

        return X, y

    def stack_data(self, list_files):
        
        X, y = [], []
        for file in list_files:
            print(f"Processing file: {file} to extract {self.frac_samples * 100:.2f}% of the data")
            X_, y_ = self.make_subset_data(file)
            X.append(X_)
            y.append(y_)

        X = np.vstack(X)
        y = np.vstack(y)

        return X, y

    def get_list_rdn_files(self):
        return self.list_rdn_files

    def get_list_all_files(self):
        return self.get_files()

    def set_list_rdn_files(self, nb_file2merge):
        self.list_rdn_files = np.random.choice(self.list_all_files, nb_file2merge, replace=False)
        self.nb_file2merge = nb_file2merge

    def load_dataset_mat(self, datafile, outfmt="torch", with_coord=False):
        file_path = os.path.join(self.data_dir, datafile)

        with h5py.File(file_path, 'r') as f:
            input_data = np.array(f.get('input')).astype(np.float32).T
            target_data = np.array(f.get('target')).astype(np.float32).T
            if with_coord:
                lat_data = np.array(f.get('latS')).astype(np.float32).T
                lon_data = np.array(f.get('lonS')).astype(np.float32).T
                input_data = np.concatenate((input_data, lat_data, lon_data), axis=1)
        if outfmt == "torch":
            return torch.from_numpy(input_data), torch.from_numpy(target_data)     
        elif outfmt == "pandas":
            return pd.DataFrame(input_data), pd.DataFrame(target_data)
        else:
            return input_data, target_data

    def save_dataset_mat(self, datafile, X, y):
        file_path = os.path.join(self.data_dir, datafile)
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('input', data=X.T)
            f.create_dataset('target', data=y.T)
            

        
    def get_files(self, directory=None, pattern=r"^trainingData.*\.mat$"):
        if directory is None:
            directory = self.data_dir
        files = [file for file in os.listdir(directory) if re.match(pattern, file)]
        self.list_all_files = files
        return files

    def box_cox_transform(self, X, fields_to_transform):
        for field in fields_to_transform:
            if any(X[field] <= 0):
                X[field] = X[field] - X[field].min() + 1
            X[field], _ = boxcox(X[field])
        return X

    def load_data(self):
        
        if self.nb_file2merge == 0:
            self.nb_file2merge = len(self.list_all_files)
         
        print(f"Number of files to merge: {self.nb_file2merge}")
        print(f"List of random files: {self.list_rdn_files}")
        if self.nb_file2merge == 0:
            raise ValueError("Please set the number of files to merge")

        if len(self.list_rdn_files) != 0:
            X, y = self.stack_data(self.list_rdn_files)
            return pd.DataFrame(X), pd.DataFrame(y)

    def normalize_data(self, X: pd.DataFrame, fields_to_transform: list) -> pd.DataFrame:
        X_boxcox = self.box_cox_transform(X.copy(), fields_to_transform)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_boxcox)
        return pd.DataFrame(X_scaled, columns=X.columns)