# Contact Task
python test_baselines.py \
    --model_type resnet \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_ResNet/bs32_ep100_lr5e-06/2025-09-23-13-04-32/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

python test_baselines.py \
    --model_type vit \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_ViT/bs32_ep100_lr5e-06/2025-09-23-12-45-54/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test


python test_baselines.py \
    --model_type convnext \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_ConvNeXt/bs32_ep100_lr5e-06/2025-09-23-13-06-05/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test


python test_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_Swin/bs32_ep100_lr5e-06/2025-09-23-13-10-31/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test


# Spatial Task
python test_baselines.py \
    --model_type resnet \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/resnet-50 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ResNet/bs32_ep100_lr5e-06/2025-09-23-13-06-54/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test

python test_baselines.py \
    --model_type vit \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/vit-base-patch16-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ViT/bs32_ep100_lr5e-06/2025-09-23-12-46-12/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test


python test_baselines.py \
    --model_type convnext \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/convnext-tiny-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ConvNeXt/bs32_ep100_lr5e-06/2025-09-23-13-10-08/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test


python test_baselines.py \
    --model_type swin \
    --pretrained_path /mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224 \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin/bs32_ep100_lr5e-06/2025-09-23-13-14-18/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test