import torch
from timm import create_model
from torch import nn


# Vision Transformer model class
class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.vit = create_model('vit_small_patch16_224', pretrained=True)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)