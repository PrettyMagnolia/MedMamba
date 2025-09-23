# Contact Task
# python train_baselines.py \
#     --model_type resnet \
#     --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
#     --model_name Contact_Task_ResNet \
#     --num_classes 2 \
#     --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train \
#     --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/val \
#     --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

# python train_baselines.py \
#     --model_type vit \
#     --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
#     --model_name Contact_Task_ViT \
#     --num_classes 2 \
#     --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train \
#     --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/val \
#     --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

# python train_baselines.py \
#     --model_type convnext \
#     --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
#     --model_name Contact_Task_ConvNeXt \
#     --num_classes 2 \
#     --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train \
#     --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/val \
#     --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

# python train_baselines.py \
#     --model_type swin \
#     --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
#     --model_name Contact_Task_Swin \
#     --num_classes 2 \
#     --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train \
#     --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/val \
#     --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

# Spatial Task
# python train_baselines.py \
#     --model_type resnet \
#     --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
#     --model_name Spatial_Task_ResNet \
#     --num_classes 4 \
#     --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train \
#     --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val \
#     --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test

# python train_baselines.py \
#     --model_type vit \
#     --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
#     --model_name Spatial_Task_ViT \
#     --num_classes 4 \
#     --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train \
#     --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val \
#     --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test

# python train_baselines.py \
#     --model_type convnext \
#     --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
#     --model_name Spatial_Task_ConvNeXt \
#     --num_classes 4 \
#     --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train \
#     --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val \
#     --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test

python train_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Spatial_Task_Swin \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test


