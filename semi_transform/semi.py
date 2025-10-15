from semi_transform.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, root, mode, size=None, nsample=None, id_path=None):
        self.root = root    # root path of the dataset
        self.mode = mode
        self.size = size
        
        if mode == 'train_l' or mode == 'train_u':
            if id_path is not None:
                with open(id_path, 'r') as f:
                    self.ids = sorted(f.read().splitlines())
            else:
                self.ids = sorted(os.listdir(os.path.join(root, 'JPEGImages')))
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            self.ids = sorted(os.listdir(os.path.join(root, 'JPEGImages')))

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, 'JPEGImages', id)).convert('RGB')
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = np.array(Image.open(os.path.join(self.root, 'SegmentationClass', id.replace('.jpg', '.png'))).convert("L"))
            mask[mask > 0] = 1
            mask = Image.fromarray(mask)

        if self.mode == 'val':
            img, mask = normalize_val(img, mask, self.size)
            return img, mask, id.replace('.jpg', '.png')

        img, mask = resize_and_crop(img, mask, 0.25, self.size)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask, self.mode)
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.1)(img_s1)
        img_s1 = blur(img_s1, p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.1)(img_s2)
        img_s2 = blur(img_s2, p=0.5)

        return normalize(img_w), normalize(img_s1), normalize(img_s2)

    def __len__(self):
        return len(self.ids)
