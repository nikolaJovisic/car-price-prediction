import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import squeeze, transpose


class ColorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(6, 6)
        self.fc = nn.Linear(48, 18)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x
