import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from PIL import Image


class CNN(nn.Module):
    def __init__(self, flatten_size: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input channel is 1 for grayscale images
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(256)

        # Calculate the flattened size
        # After four pooling layers, the size is reduced to 128 / (2^3) = 16 in each dimension
        self.flattened_size = flatten_size # feature * h_dim * w_dim

        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)  # 10 classes

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = self.pool(x)
        x = f.relu(self.conv4(x))
        x = self.bn(x)
        x = f.dropout(x, 0.25)
        x = torch.flatten(x, 1)  # flatten
        x = f.relu(self.fc1(x))
        x = f.sigmoid(self.fc2(x))
        x = f.dropout(x, 0.5)
        x = f.softmax(self.fc3(x), dim=1)
        return x
