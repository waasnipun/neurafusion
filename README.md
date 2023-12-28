# Neurafusion

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [How to setup](#how-to-setup)
    - [Usage](#usage)
        - [Pre-training](#pre-training)
        - [Fine-tuning](#fine-tuning)
        - [Visualization](#visualization)
3. [Results](#results)

## Introduction

Neurafusion is a cross-domain few-shot learning project perform using various pre-trained models such as Vision Transformers (ViT), EfficientNet, and ResNet18. The main model that primary emphasis is on Vision Transformer(ViT). The project includes tools for pre-training and fine-tuning models on different datasets, specifically MiniImageNet and EuroSAT_RGB. Please follow the below instructions to get started

## Getting Started

Follow these steps to get started with the project:

### How to setup

1. Install the required dependencies:

    ```bash
    Python 3.9.18
    Torch 2.1.1
    Torchvision 0.16.1
    Timm 0.9.12
    Matplotlib
    Numpy
    Sklearn
   ```

2. Copy the MiniImageNet dataset and EuroSat dataset to datasets directory to the root directory. Directory structure should look like this

```
├── datasets 
│   ├── miniImageNet 
│   │   ├── n01532829
│   │   .
│   │   .
│   │   .
│   │   └── n13133613
│   └── EuroSAT_RGB
│       ├── AnnualCrop
│       .
│       .
│       .
│       └── SeaLake
│
├── images
│
├── models
│   ├── ViT.py
│   ├── efficientnet.py
│   └── resnet18.py
│
├── pretrain-ViT.ipynb 
├── pretrain-efficientnet.ipynb 
├── pretrain-resnet18.ipynb 
│
├── pretrained 
│   ├── efficientnet_model_best.pth
│   ├── resnet_model_best.pth
│   └── vit_model_best.pth
│
├── requirement.txt
├── tools.py
│
├── train-ViT.ipynb 
├── train-efficientnet.ipynb 
├── train-resnet18.ipynb 
│
├── trained 
│   ├── efficientnet_trained.pth
│   ├── resnet18.pth
│   └── vit_trained.pth
│
├── visualize_pretrain.ipynb 
└── visualize_train.ipynb 
```

**Note:** Extract the class folders from `train.tar`, `val.tar` and `test.tar` into a single miniImageNet datasets folder like above so that it contains 100 folders representing 100 classes. The filepaths are setted up for the datasets folder already.

### Usage

1. **Pre-training:**
    - Open the ViT pretrain notebook (e.g., `pretrain-ViT.ipynb`).
    - Run the notebook to pre-train the selected model on the MiniImageNet dataset.

2. **Fine-tuning:**
    - Open the ViT train notebook (e.g., `train-ViT.ipynb`).
    - Run the notebook to fine-tune a pre-trained model on a EuroSAT dataset.

3. **(Optional)Visualization of the datasetsets:**
    - Explore the visualization notebooks (`visualize_EuroSAT.ipynb` and `visualize_miniImageNet.ipynb`) to understand the training and pre-training processes and distributions.


### Results

ViT results on evaluating against euroSAT dataset

![Image](/images/vit_euroset.png)

Jupyter notebooks for visualizing fine-tuning dataset.