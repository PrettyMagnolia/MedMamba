import os
import shutil
import random

def split_dataset(root_dir, output_dir, task, train_ratio=0.6, val_ratio=0.2, seed=42):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    random.seed(seed)
    if task == 'contact':
        classes = ['contact', 'not contact']
    elif task == 'spatial':
        classes = ['apical', 'between the roots', 'buccal', 'lingual']
    else:
        classes = ['between the roots', 'not between the roots']

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        # if not os.path.isdir(cls_dir):
        #     continue
        if cls == 'not between the roots':
            sub_cls = ['apical', 'buccal', 'lingual']
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

        if cls == 'not between the roots':
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
    # root_folder = '/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img_new'
    # output_folder = '/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_new'
    # split_dataset(root_folder, output_folder, 'contact')

    root_folder = '/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img'
    output_folder = '/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task'
    split_dataset(root_folder, output_folder, 'spatial')

    # root_folder = '/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img'
    # output_folder = '/home/yifei/code/Med_CV/MedMamba/dataset/5_spatial_task_mid'
    # split_dataset(root_folder, output_folder, 'spatial_mid')