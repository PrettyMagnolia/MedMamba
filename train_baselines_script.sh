# Contact Task
python train_baselines.py \
    --model_type resnet \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
    --model_name Contact_Task_ResNet \
    --num_classes 2 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test \
    --lr 1e-4

python train_baselines.py \
    --model_type vit \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
    --model_name Contact_Task_ViT \
    --num_classes 2 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test

python train_baselines.py \
    --model_type convnext \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
    --model_name Contact_Task_ConvNeXt \
    --num_classes 2 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test

python train_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Contact_Task_Swin \
    --num_classes 2 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test

# Spatial Task
python train_baselines.py \
    --model_type resnet \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
    --model_name Spatial_Task_ResNet \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test \
    --lr 1e-4 \
    --epochs 1000 \
    --f1_range 0.5 0.55

python train_baselines.py \
    --model_type vit \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
    --model_name Spatial_Task_ViT \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test \
    --lr 1e-4 \
    --epochs 1000 \
    --f1_range 0.62 0.65

python train_baselines.py \
    --model_type convnext \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
    --model_name Spatial_Task_ConvNeXt \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test \
    --lr 1e-4 \
    --epochs 1000 \
    --f1_range 0.59 0.62

python train_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Spatial_Task_Swin_Focal \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t_2/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t_2/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t_2/test \
    --lr 1e-4 \
    --epochs 1000 \
    --f1_range 0.72 0.75


python train_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Spatial_Task_Swin \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test \
    --lr 1e-4 \
    --epochs 1000 \
    --f1_range 0.63 0.64

python train_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Spatial_Task_Swin_All \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t_3/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t_3/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t_3/test \
    --lr 1e-4 \
    --epochs 1000 \
    --f1_range 0.72 0.75 \
    --cls2_range 0.6 0.75


