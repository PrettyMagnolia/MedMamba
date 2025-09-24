python train_medmamba.py \
    --model_name Contact_Task_MedMamba \
    --num_classes 2 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test

python train_medmamba.py \
    --model_name Spatial_Task_MedMamba \
    --num_classes 4 \
    --train_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train \
    --val_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val \
    --test_root_dir /home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test