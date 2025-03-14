# Deep Learning based Image Denoising Models

## About the project

### Project structure
```
.
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── denoising_cae.py
│   │   ├── denoising_cnn.py
│   │   ├── denoising_unet.py
│   │   └── model_select.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── learning.py
│   │   └── metrics.py
│   ├── test.py
│   └── train.py
├── README.md
└── requirements.txt
```
## Getting started

## Experiment

### Procedure

### Results
denoising performance

            no batchnorm                        batchnorm                   batchnorm+harder

CNN     PSNR: 24.5342, SSIM: 0.7614 PSNR: 24.0136, SSIM: 0.7426   PSNR: 20.8161, SSIM: 0.5922

CAE     PSNR: 19.6813, SSIM: 0.5636 PSNR: 21.5009, SSIM: 0.6647   PSNR: 19.6031, SSIM: 0.5732

UNET    PSNR: 23.7105, SSIM: 0.7328 PSNR: 24.4413, SSIM: 0.7610   PSNR: 21.3482, SSIM: 0.6338
