import torch

class Config:
    # Dataset configuration
    dataset_root = "./data"
    batch_size = 64
    num_workers = 4
    
    # Model configuration
    num_classes = 10
    
    # Training configuration
    num_epochs = 10
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"