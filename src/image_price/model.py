import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(64, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def get_resnet18():
    model = resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # for param in model.layer4.parameters():
        # param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def get_model():
    # return PricePredictor()
    return get_resnet18()
