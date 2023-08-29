import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

# Path to store the LMDB
lmdb_path = 'cifar10_lmdb'
os.makedirs(lmdb_path, exist_ok=True)

# Transform to convert PIL image to numpy array
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Open LMDB environment
env = lmdb.open(lmdb_path, map_size=int(1e9))

# Function to serialize and store data in LMDB
def store_in_lmdb(dataset, prefix):
    with env.begin(write=True) as txn:
        for i, (data, target) in tqdm(enumerate(dataset), total=len(dataset), desc=prefix):
            key = f'{prefix}_{i:05}'.encode('utf-8')
            datum = {'data': data.numpy(), 'target': target}
            txn.put(key, pickle.dumps(datum))

# Store training data in LMDB
store_in_lmdb(train_dataset, 'train')

# Store test data in LMDB
store_in_lmdb(test_dataset, 'test')

# Close LMDB environment
env.close()

print("LMDB creation completed.")