# Contact Task
python test_baselines_v2.py \
    --model_type resnet \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_ResNet/bs32_ep50_lr0.0001/2025-10-20-08-24-41/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test

python test_baselines_v2.py \
    --model_type vit \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_ViT/bs32_ep50_lr5e-06/2025-10-20-08-07-57/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test


python test_baselines_v2.py \
    --model_type convnext \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_ConvNeXt/bs32_ep50_lr5e-06/2025-10-19-14-19-40/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test


python test_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_Swin/bs32_ep50_lr5e-06/2025-10-18-09-54-07/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test


# Spatial Task
python test_baselines_v2.py \
    --model_type resnet \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ResNet/bs64_ep1000_lr0.0001/2025-11-07-07-15-01/last.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test

python test_baselines_v2.py \
    --model_type vit \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ViT/bs64_ep1000_lr0.0001/2025-10-23-09-07-37/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test


python test_baselines_v2.py \
    --model_type convnext \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ConvNeXt/bs64_ep1000_lr0.0001/2025-11-07-08-52-44/last.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test


python test_baselines_v2.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin/bs64_ep1000_lr0.0001/2025-11-07-08-02-25/last.pth\
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test

# focal loss + over sampling
python test_baselines_v2.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_Over/bs64_ep1000_lr0.0001/2025-10-23-10-43-38/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test

python test_baselines_v2.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_Focal/bs64_ep1000_lr0.0001/2025-10-25-03-49-00/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test

python test_baselines_v2.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-06-14/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test


# 集成学习测试
python test_ensemble.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-32-00/last.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test

# 集成学习测试（三分类）
python test_ensemble_three.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-32-00/last.pth \
    --num_classes 3 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test