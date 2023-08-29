import torch
import torch.nn as nn

def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)
            
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()
    
    accuracy = total_correct / total_samples
    return accuracy

# Assuming you have already defined the model, criterion, optimizer, etc.
# Load the model weights (if needed) and move the model to the device
model.load_state_dict(torch.load('model_weights.pth'))
model.to(device)

# Evaluate the model on the test dataset
test_accuracy = evaluate_model(model, test_loader_custom, device)
print(f'Test Accuracy: {test_accuracy:.4f}')
