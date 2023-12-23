import torch.nn as nn
from timm import create_model

# Define ResNet10 model
class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        self.efficientNet = create_model('efficientnet_b0', pretrained=True)
        in_features = self.efficientNet.classifier.in_features
        self.efficientNet.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.efficientNet(x)