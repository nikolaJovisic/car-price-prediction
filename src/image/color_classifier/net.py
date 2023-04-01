import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import squeeze, transpose


class ColorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 5)
        self.pool = nn.MaxPool2d(4, 6)
        self.fc = nn.Linear(16, 18)

    def forward(self, x):
        image_disp = squeeze(x[0])
        image_disp = transpose(image_disp, 0, 2)
        plt.imshow(image_disp)
        plt.show()

        x = F.relu(self.conv(x))

        image_disp = x[0].detach().numpy()
        image_disp = np.transpose(image_disp, [1, 2, 0])
        plt.imshow(image_disp)
        plt.show()

        x = self.pool(x)

        image_disp = x[0].detach().numpy()
        image_disp = np.transpose(image_disp, [1, 2, 0])
        plt.imshow(image_disp)
        plt.show()

        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x
