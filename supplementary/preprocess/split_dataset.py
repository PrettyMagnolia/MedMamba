import os
import shutil
import random
import argparse
import json


def split_dataset(root_dir, output_dir, classes, train_ratio=0.6, val_ratio=0.2, seed=42, merge_groups=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    random.seed(seed)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        if merge_groups and cls in merge_groups:
            sub_cls = merge_groups[cls]
            images = []
            for sub in sub_cls:
                sub_dir = os.path.join(root_dir, sub)
                if os.path.isdir(sub_dir):
                    sub_images = [f for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f))]
                    images.extend([os.path.join(sub, f) for f in sub_images])
        else:
            images = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]

        random.shuffle(images)
        n = len(images)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        train_cls_dir = os.path.join(train_dir, cls)
        val_cls_dir = os.path.join(val_dir, cls)
        test_cls_dir = os.path.join(test_dir, cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        os.makedirs(test_cls_dir, exist_ok=True)

        if merge_groups and cls in merge_groups:
            for img in train_images:
                shutil.copy(os.path.join(root_dir, img), os.path.join(train_cls_dir, os.path.basename(img)))
            for img in val_images:
                shutil.copy(os.path.join(root_dir, img), os.path.join(val_cls_dir, os.path.basename(img)))
            for img in test_images:
                shutil.copy(os.path.join(root_dir, img), os.path.join(test_cls_dir, os.path.basename(img)))
        else:
            for img in train_images:
                shutil.copy(os.path.join(cls_dir, img), os.path.join(train_cls_dir, img))
            for img in val_images:
                shutil.copy(os.path.join(cls_dir, img), os.path.join(val_cls_dir, img))
            for img in test_images:
                shutil.copy(os.path.join(cls_dir, img), os.path.join(test_cls_dir, img))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing class folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for train/val/test splits')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='List of class names')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--merge_groups', type=str, default=None, help='JSON mapping: merged class -> list of source classes, e.g. \'{"not between the roots": ["apical", "buccal", "lingual"]}\'')

    args = parser.parse_args()

    merge_groups = None
    if args.merge_groups:
        merge_groups = {k: v for k, v in json.loads(args.merge_groups).items()}

    split_dataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        merge_groups=merge_groups
    )
