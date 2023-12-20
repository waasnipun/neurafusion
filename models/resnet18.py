import torch.nn as nn
from torchvision.models import resnet18, resnet50

# Define ResNet10 model
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18(weights='ResNet18_Weights.DEFAULT')
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)