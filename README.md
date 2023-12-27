# Neurafusion

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Usage](#usage)
        - [Pre-training](#pre-training)
        - [Fine-tuning](#fine-tuning)
        - [Visualization](#visualization)
    - [Directory Structure](#directory-structure)

## Introduction

Neurafusion is a cross-domain few-shot learning project perform using various pre-trained models such as Vision Transformers (ViT), EfficientNet, and ResNet18. The project includes tools for pre-training and fine-tuning models on different datasets, specifically MiniImageNet and EuroSAT_RGB.

## Getting Started

Follow these steps to get started with the project:

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/waasnipun/neurafusion.git
    cd neurafusion
    ```

2. Install the required dependencies:

    ```bash
    Python 3.9.18
    Torch 2.1.1
    Torchvision 0.16.1
    Timm 0.9.12
    Matplotlib
    Numpy
    ```

### Usage

1. **Pre-training:**
    - Open the relevant pretrain notebook (e.g., `pretrain-ViT.ipynb`).
    - Run the notebook to pre-train the selected model on the MiniImageNet dataset.

2. **Fine-tuning:**
    - Open the relevant train notebook (e.g., `train-ViT.ipynb`).
    - Run the notebook to fine-tune a pre-trained model on a EuroSAT dataset.

3. **Visualization:**
    - Explore the visualization notebooks (`visualize_EuroSAT.ipynb` and `visualize_miniImageNet.ipynb`) to understand the training and pre-training processes.

### Directory Structure

```
├── datasets 
│   ├── miniImageNet 
│   │   ├── n01532829
│   │   .
│   │   .
│   │   .
│   │   └── n13133613
│   └── EuroSAT_RGB
│   │   ├── AnnualCrop
│   │   .
│   │   .
│   │   .
│   │   └── SeaLake
│
├── images
│
├── models - Implementation of different models.
│   ├── ViT.py
│   ├── efficientnet.py
│   └── resnet18.py
│
├── pretrain-ViT.ipynb - Jupyter notebooks for pre-training Vision Transformer model.
├── pretrain-efficientnet.ipynb - Jupyter notebooks for pre-training Efficientnet model.
├── pretrain-resnet18.ipynb - Jupyter notebooks for pre-training Resnet18 model.
│
├── pretrained - Pre-trained weights for the models.
│   ├── efficientnet_model_best_78.64.pth
│   ├── resnet_model_best_76.89.pth
│   └── vit_model_best_89.84.pth
│
├── requirement.txt
├── tools.py
│
├── train-ViT.ipynb - Jupyter notebooks for fine-tuning Vision Transformer model.
├── train-efficientnet.ipynb - Jupyter notebooks for fine-tuning Efficientnet model.
├── train-resnet18.ipynb - Jupyter notebooks for fine-tuning Resnet18 model.
│
├── trained - Fine-tuned weights after training.
│   ├── efficientnet_trained_60.13.pth
│   ├── resnet18_68.75.pth
│   └── vit_trained_74.0.pth
│
├── visualize_pretrain.ipynb - Jupyter notebooks for visualizing pre-training dataset.
└── visualize_train.ipynb - Jupyter notebooks for visualizing fine-tuning dataset.
```
