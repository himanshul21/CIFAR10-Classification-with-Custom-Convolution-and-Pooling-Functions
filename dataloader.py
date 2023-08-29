import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, prefix, transform=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.prefix = prefix
        self.transform = transform

    def __len__(self):
        with self.env.begin(write=False) as txn:
            return txn.stat()['entries']

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = f'{self.prefix}_{idx:05}'.encode('utf-8')
            value = txn.get(key)

        datum = pickle.loads(value)
        data = datum['data']
        target = datum['target']

        if self.transform:
            data = self.transform(data)

        return data, target