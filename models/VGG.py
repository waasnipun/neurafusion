import torch.nn as nn
from torchvision.models import vgg

# Define ResNet10 model
class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.vgg = vgg(pretrained=False)
        in_features = self.vgg.fc.in_features
        self.vgg.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)