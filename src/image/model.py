import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ColorClassifier(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.pool = nn.MaxPool2d(6, 6)
        self.fc1 = nn.Linear(27, 24)
        self.fc2 = nn.Linear(24, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_resnet18(outputs_size):
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, outputs_size)
    return model

def get_model(outputs_size):
    # return ColorClassifier(outputs_size)
    return get_resnet18(outputs_size)
