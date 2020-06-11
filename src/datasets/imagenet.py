import os
import copy
import json
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

if 'ubuntu-156184-1' in os.uname()[1]:
    IMAGENET_DIR = '/data2/wumike/Dataset/imagenet_raw'
else:
    IMAGENET_DIR = '/data5/chengxuz/Dataset/imagenet_raw'


class ImageNet(data.Dataset):
    def __init__(self, train=True, imagenet_dir=IMAGENET_DIR, image_transforms=None):
        super().__init__()
        split_dir = 'train' if train else 'validation'
        self.imagenet_dir = os.path.join(imagenet_dir, split_dir)
        self.dataset = datasets.ImageFolder(self.imagenet_dir, image_transforms)
    
    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # important to return the index!
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)
