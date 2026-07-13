# Supplementary Code

This directory contains the cleaned code used in our experiments.

```
supplementary/
├── preprocess/
│   ├── data_augment.py          # Data augmentation
│   └── split_dataset.py         # Dataset splitting
├── test/
│   ├── test_models.py           # Single model testing (per-class ROC/PR plots)
│   ├── test_models_v2.py        # Single model testing (macro-averaged CSV export)
│   ├── test_ensemble.py         # Two-model ensemble testing
│   └── test_ensemble_three.py   # Three-model ensemble testing
├── train/
│   └── train_models.py          # Model training (ResNet/ViT/ConvNeXt/Swin)
└── visualization/
    ├── grad_cam.py               # Grad-CAM interpretability visualization
    ├── plot.py                   # Multi-model PR/ROC curve comparison plots
    └── tsne.py                   # t-SNE feature dimensionality reduction visualization
```
