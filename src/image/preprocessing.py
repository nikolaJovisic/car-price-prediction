import math

import torch
from kornia.color import RgbToHsv
from torchvision import transforms
from torchvision.transforms import Lambda, Resize


def scale_hue(image):
    """
    RgbToHsv in kornia package scales hue to [0, 2pi],
    here is scaled to [0, 1].
    """
    image[0] /= 2 * math.pi
    return image


preprocessing = transforms.Compose(
    [
        Resize((32, 32)),
        transforms.ConvertImageDtype(torch.float),
        RgbToHsv(),
        Lambda(lambd=scale_hue),
    ]
)
