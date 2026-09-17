import os
import glob
import pandas as pd
import numpy as np
import pyproj
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA

class IOVNBDDataset(Dataset):
    def __init__(self, data_dir="data/raw/IO-VNBD", window_size=20):
        self.window_size = window_size
        self.samples = []
        self.labels = []
        self.transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:32643", always_xy=True)
        
        search_pattern = os.path.join(data_dir, "**", "S-Dataset", "*.csv")
        file_paths = glob.glob(search_pattern, recursive=True)
        
        for file in file_paths:
            self._process_file(file)
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def _process_file(self, file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception:
            return
            
        # Dynamically locate columns regardless of exact AndroSensor version
        cols = [c.upper() for c in df.columns]
        accel_cols = [df.columns[i] for i, c in enumerate(cols) if 'ACCELEROMETER' in c or 'ACCEL' in c][:3]
        gyro_cols = [df.columns[i] for i, c in enumerate(cols) if 'GYROSCOPE' in c or 'GYRO' in c][:3]
        lat_col = [df.columns[i] for i, c in enumerate(cols) if 'LATITUDE' in c or 'LAT' in c][0]
        lon_col = [df.columns[i] for i, c in enumerate(cols) if 'LONGITUDE' in c or 'LON' in c][0]
        
        if len(accel_cols) < 3 or len(gyro_cols) < 3:
            return

        features = df[accel_cols + gyro_cols].ffill().bfill().fillna(0).values
        if len(features) <= self.window_size:
            return
            
        # Align arbitrary smartphone mounting to vehicle forward axis
        pca = PCA(n_components=3)
        features[:, 0:3] = pca.fit_transform(features[:, 0:3])
        
        x, y = self.transformer.transform(df[lon_col].values, df[lat_col].values)
        delta_x = np.diff(x, prepend=x[0])
        delta_y = np.diff(y, prepend=y[0])
        
        for i in range(len(features) - self.window_size):
            window = features[i : i + self.window_size]
            target_dx = np.sum(delta_x[i : i + self.window_size])
            target_dy = np.sum(delta_y[i : i + self.window_size])
            
            self.samples.append(window)
            self.labels.append([target_dx, target_dy])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]