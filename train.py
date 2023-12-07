import torch
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from mini_imagenet_dataset import MiniImageNetDataset

batch_size = 64
num_workers = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")
print('Device:', device)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MiniImageNetDataset(root_dir=os.getcwd(), phase='train', shuffle_images=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = MiniImageNetDataset(root_dir=os.getcwd(), phase='val', shuffle_images=True, transform=transform)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataset = MiniImageNetDataset(root_dir=os.getcwd(), phase='test', shuffle_images=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

