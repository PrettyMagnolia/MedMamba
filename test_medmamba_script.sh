# Contact Task
python test_medmamba.py \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_MedMamba/bs32_ep100_lr5e-06/2025-09-24-08-11-22/best.pth \
    --num_classes 2 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

# Spatial Task
python test_medmamba.py \
    --ckpt_path /home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_MedMamba/bs32_ep100_lr5e-06/2025-09-24-08-15-39/best.pth \
    --num_classes 4 \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test