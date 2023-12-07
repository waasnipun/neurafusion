import torch.nn as nn
from torchvision.models import vgg, vgg16


# Define ResNet10 model
class VGG(nn.Module):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.vgg = vgg16(pretrained=False)  # You can choose a different variant (e.g., vgg16, vgg19)

        # Check the last layer of the classifier
        if isinstance(self.vgg.classifier[6], nn.Linear):
            in_features = self.vgg.classifier[6].in_features
            self.vgg.classifier[6] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unexpected classifier layer type, modify as needed.")

    def forward(self, x):
        return self.vgg(x)