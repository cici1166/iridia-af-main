import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ECGH5Dataset(Dataset):
    """
    读取 processed_data.h5 里的数据集
    支持CSV格式：sample_id,segment_label
    """

    def __init__(self, h5_path: str, csv_path: str, normalize: bool = True):
        self.h5_path = h5_path
        self.csv = pd.read_csv(csv_path)
        self.normalize = normalize
        self._h5 = None  # 懒加载，避免多进程句柄冲突

    def _open_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, i):
        row = self.csv.iloc[i]
        sample_id = row['sample_id']
        label = 1 if row['segment_label'] == 'pre_af' else 0
        
        h5 = self._open_h5()
        sig = np.array(h5[sample_id], dtype=np.float32)  # (T,)
        
        if self.normalize:
            std = sig.std() + 1e-6
            sig = (sig - sig.mean()) / std
        x = torch.from_numpy(sig).unsqueeze(0)  # (1, T)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass 