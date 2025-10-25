import albumentations as A
import torch
import math
import random
import os
import shutil
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image

# dataset_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train'
# dataset_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train'
dataset_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train'

seed = 42
level = 5
times = 4

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)

def odd_conversion(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num

transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        # A.Posterize(num_bits=max(1, int(8 - 0.8 * level)), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, odd_conversion(3 + 0.8 * level)), p=0.2 * level),
        # A.GaussNoise(var_limit=(1e-4 * level, 1e-5 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, fill=0, rotate_method='largest_box', crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=0, shear=0, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent=None, translate_px=None, rotate=0, shear={'x': (0, 2 * level), 'y': (0, 0)}, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent=None, translate_px=None, rotate=0, shear={'x': (0, 0), 'y': (0, 2 * level)}, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=0, shear=0, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=0, shear=0, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level)
    ])

strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]

def med_augment(file_path, times=times):
    output_dir = os.path.dirname(file_path)
    ori_image_name = file_path.split('/')[-1].split('.')[0]

    for i in range(times):
        employ = random.choice(strategy)
        pixel, shape = random.sample(transform[:4], employ[0]), random.sample(transform[4:], employ[1])
        img_transform = A.Compose([*pixel, *shape])
        # img_transform = A.Compose(transform[5:6])
        random.shuffle(img_transform.transforms)

        image = np.array(Image.open(file_path))
        augmented = img_transform(image=image)
        augmented_image = Image.fromarray(augmented['image'])
        augmented_image.save(os.path.join(output_dir, f"{ori_image_name}_aug_{i}.png"))

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            file_path = os.path.join(root, file)
            med_augment(file_path, times=4)
            print('process image: {}'.format(file))
