import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None, mode=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        if mode == 'train_l':
            mask = transforms.ToTensor()(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def normalize_val(img, mask, size):
    img = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    mask = torch.from_numpy(np.array(mask)).long()
    return img, mask


def resize_and_crop(img, mask, ratio_range, crop_size):
    # ## resize
    new_w = random.randint(int(crop_size * (1 - ratio_range)), int(crop_size * (1 + ratio_range)))
    new_h = random.randint(int(crop_size * (1 - ratio_range)), int(crop_size * (1 + ratio_range)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    mask = mask.resize((new_w, new_h), Image.NEAREST)

    sizew_forresize = new_w if new_w >= crop_size else crop_size
    sizeh_forresize = new_h if new_h >= crop_size else crop_size
    img = img.resize((sizew_forresize, sizeh_forresize), Image.BILINEAR)
    mask = mask.resize((sizew_forresize, sizeh_forresize), Image.NEAREST)

    # ## crop
    x = random.randint(0, sizew_forresize - crop_size)
    y = random.randint(0, sizeh_forresize - crop_size)
    img = img.crop((x, y, x + crop_size, y + crop_size))
    mask = mask.crop((x, y, x + crop_size, y + crop_size))

    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img
