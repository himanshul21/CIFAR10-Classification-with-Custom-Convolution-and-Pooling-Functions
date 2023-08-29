import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride = 1):
        super(Conv2D, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize weights using nn.init module
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size, in_channels, height, width = x.size()
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Apply padding to the input
        x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        
        # Extract patches from the padded input image
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches = patches.contiguous().view(batch_size, in_channels, -1, self.kernel_size, self.kernel_size)

        # Perform element-wise multiplication and sum across kernel dimensions
        conv_out = (patches.unsqueeze(1) * self.weight.unsqueeze(2)).sum([2, 4, 5])

        # Add bias and reshape to output shape
        conv_out = conv_out + self.bias.view(1, -1, 1)
        
        conv_out = conv_out.view(batch_size, -1, out_height, out_width)
        
        return conv_out

class MaxPooling2D(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculate output dimensions
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        
        # Reshape the input tensor to prepare for vectorized max pooling
        
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches = patches.contiguous().view(batch_size, channels, -1, self.kernel_size, self.kernel_size)

        # Reshape the unfolded tensor to match the output dimensions
        unfolded_output = patches.view(batch_size, channels, output_height, output_width, -1)
        
        # Take the maximum value along the kernel_size * kernel_size dimension
        output, _ = unfolded_output.max(dim=-1)
        
        return output

class AvgPooling2D(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(AvgPooling2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Calculate output dimensions
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        
        # Reshape the input tensor to prepare for vectorized max pooling
        patches = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        patches = patches.contiguous().view(batch_size, channels, -1, self.kernel_size, self.kernel_size)

        # Reshape the unfolded tensor to match the output dimensions
        unfolded_output = patches.view(batch_size, channels, output_height, output_width, -1)
        
        # Take the maximum value along the kernel_size * kernel_size dimension
        output = unfolded_output.mean(dim=-1)
        
        return output

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        
        # Initialize weights using nn.init module
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.bias)
        
    def forward(self, x):
        # Vectorized implementation
        output = torch.mm(x, self.weight.t()) + self.bias.unsqueeze(0)
        
        return output

class Dropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full(x.size(), 1 - self.p)).to(device)
            return x * mask / (1 - self.p)
        else:
            return x

class ReLU(nn.Module):
    def forward(self, x):
        output = torch.max(x, torch.tensor(0.0))
        return output

class Sigmoid(nn.Module):
    def forward(self, x):
        output = 1 / (1 + torch.exp(-x))
        return output

class Softmax(nn.Module):
    def forward(self, x):
        exp_vals = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])  # Subtracting max for numerical stability
        output = exp_vals / torch.sum(exp_vals, dim=1, keepdim=True)
        return output