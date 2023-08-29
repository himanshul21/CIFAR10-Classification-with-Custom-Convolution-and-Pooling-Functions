import torch
from torch import optim
from model import AlexNet #Model
from dataloader import LMDBDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from loss import CrossEntropyLoss
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import copy
# from config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# batch_size = 4

# Load LMDBDataset for training
train_dataset = LMDBDataset(lmdb_path='cifar10_lmdb', prefix='train', transform = transform)

# Create DataLoader for training
batch_size = 4
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # model = Model().to(device)
# net = AlexNet(num_classes = 10).to(device)

# criterion = CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# dataloaders = {'train' : trainloader, 'val' : testloader}
# dataset_sizes = {'train' : 50000, 'val': 10000}

# def train(model, criterion, optimizer, scheduler, num_epochs=25):
#     since = time.time()

#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0

#     for epoch in tqdm(range(num_epochs)):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs).to(device)
                    
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)
                    
# #                     print(loss)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#             if phase == 'train':
#                 scheduler.step()

#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         print()

#     time_elapsed = time.time() - since
#     print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
#     print(f'Best val Acc: {best_acc:4f}')

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model

# net = train(net, criterion, optimizer, exp_lr_scheduler)





















































# conf = Config()

# # Define data transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to [-1, 1]
# ])

# # Specify dataset paths
# train_dataset = CIFAR10Dataset(root="./data", train=True, transform=transform)
# test_dataset = CIFAR10Dataset(root="./data", train=False, transform=transform)

# # Create DataLoader instances
# batch_size = 4
# train_loader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False)

# # Define model
model = AlexNet(num_classes = 10).to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

num_epochs = 10
# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(trainloader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        # print("data", data.shape)
        outputs = model(data)
        # print(outputs, targets)
        # _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(trainloader)}] Loss: {loss.item():.4f}")

print("Training finished")