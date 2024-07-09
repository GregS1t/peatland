import os
import re
import numpy as np
import torch
import h5py
from contextlib import closing


import os
import numpy as np
import pandas as pd
import torch
import h5py

from scipy.stats import boxcox

from sklearn.preprocessing import MinMaxScaler

class PeatNetDataProc:
    def __init__(self, data_dir, frac_samples=0.3, seed=42):
        self.data_dir = data_dir
        self.frac_samples = frac_samples
        self.seed = seed

    def make_subset_data(self, datafile):
        file_path = os.path.join(self.data_dir, datafile)
        with h5py.File(file_path, 'r') as f:
            input_data = np.array(f.get('input')).astype(np.float32).T
            target_data = np.array(f.get('target')).astype(np.float32).T

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

    def load_dataset_mat(self, datafile):
        file_path = os.path.join(self.data_dir, datafile)
        with h5py.File(file_path, 'r') as f:
            input_data = np.array(f.get('input')).astype(np.float32).T
            target_data = np.array(f.get('target')).astype(np.float32).T

        X = torch.from_numpy(input_data)
        y = torch.from_numpy(target_data)

        return X, y

    def get_files(self, directory=None, pattern=r"^trainingData.*\.mat$"):
        if directory is None:
            directory = self.data_dir
        files = [file for file in os.listdir(directory) if re.match(pattern, file)]
        return files

    def box_cox_transform(self, X, fields_to_transform):
        for field in fields_to_transform:
            if any(X[field] <= 0):
                X[field] = X[field] - X[field].min() + 1
            X[field], _ = boxcox(X[field])
        return X

    def load_data(self, nb_file2merge):
        list_files = self.get_files()
        list_files_selected = np.random.choice(list_files, nb_file2merge, replace=False)
        X, y = self.stack_data(list_files_selected)
        return pd.DataFrame(X), pd.DataFrame(y)

    def normalize_data(self, X: pd.DataFrame, fields_to_transform: list) -> pd.DataFrame:
        X_boxcox = self.box_cox_transform(X.copy(), fields_to_transform)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_boxcox)
        return pd.DataFrame(X_scaled, columns=X.columns)