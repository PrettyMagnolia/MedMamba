# Contact Task
python swin_plus.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Contact_Task_Swin \
    --num_classes 2 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

# Spatial Task
python swin_plus.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Spatial_Task_Swin \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test

python test_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Mid_Swin_plus/bs256_ep50_lr1e-05/2025-10-15-07-08-10/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/5_spatial_task_mid/test


# Spatial Mid Task
python swin_plus_optuna.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Spatial_Task_Mid_Swin \
    --num_classes 2 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/5_spatial_task_mid/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/5_spatial_task_mid/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/5_spatial_task_mid/test



python ensemble.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Mid_Swin_plus/bs256_ep50_lr1e-05/2025-10-15-07-08-10/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/5_spatial_task_mid/test


python swin_plus_optuna.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --model_name Spatial_Task_New_Swin \
    --num_classes 3 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test \
    --lr 1e-4

python test_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_New_Swin_plus/bs256_ep50_lr0.0001/2025-10-15-10-21-11/best.pth \
    --num_classes 3 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test

