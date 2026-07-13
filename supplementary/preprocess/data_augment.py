import albumentations as A
import math
import random
import os
import argparse
import numpy as np
from PIL import Image


def odd_conversion(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


def build_transform(level):
    return A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, odd_conversion(3 + 0.8 * level)), p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, fill=0, rotate_method='largest_box', crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=0, shear=0, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent=None, translate_px=None, rotate=0, shear={'x': (0, 2 * level), 'y': (0, 0)}, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent=None, translate_px=None, rotate=0, shear={'x': (0, 0), 'y': (0, 2 * level)}, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=0, shear=0, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level),
        A.Affine(scale=1.0, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=0, shear=0, interpolation=1, cval=0, mode=0, fit_output=False, keep_ratio=True, p=0.2 * level)
    ])


STRATEGY = [(1, 2), (0, 3), (0, 2), (1, 1)]


def med_augment(file_path, transform, times=4):
    output_dir = os.path.dirname(file_path)
    ori_image_name = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(times):
        employ = random.choice(STRATEGY)
        pixel, shape = random.sample(transform[:4], employ[0]), random.sample(transform[4:], employ[1])
        img_transform = A.Compose([*pixel, *shape])
        random.shuffle(img_transform.transforms)

        image = np.array(Image.open(file_path))
        augmented = img_transform(image=image)
        augmented_image = Image.fromarray(augmented['image'])
        augmented_image.save(os.path.join(output_dir, f"{ori_image_name}_aug_{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing training images (with class subfolders)')
    parser.add_argument('--level', type=int, default=5, help='Augmentation intensity level')
    parser.add_argument('--times', type=int, default=4, help='Number of augmented copies per image')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    transform = build_transform(args.level)

    for root, dirs, files in os.walk(args.dataset_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                med_augment(file_path, transform, times=args.times)
                print(f'process image: {file}')
