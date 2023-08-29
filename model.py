import torch
from torch import nn
from layers import Conv2D, MaxPooling2D, AvgPooling2D, Dropout, Linear, ReLU, Softmax


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            Conv2D(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            MaxPooling2D(kernel_size=3, stride=2),
            Conv2D(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            MaxPooling2D(kernel_size=3, stride=2),
            Conv2D(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            # Conv2D(384, 384, kernel_size=3, padding=1),
            # ReLU(),
            Conv2D(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            MaxPooling2D(kernel_size=3, stride=2)
        )
        self.avgpool = AvgPooling2D(kernel_size=6, stride=1)
        self.classifier = nn.Sequential(
            # Dropout(),
            Linear(256, 512),
            nn.ReLU(),
            # Dropout(),
            Linear(512, 512),
            nn.ReLU(),
            Linear(512, num_classes)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # print("x before flatten : ", x.shape)
        x = torch.flatten(x, 1)
        # print("x after flatten : ", x.shape)
        x = self.classifier(x)
        # x = self.softmax(x)
        return x