import os
from torch.utils.data import Dataset
from PIL import Image
import random

class Dataset(Dataset):
    def __init__(self, dataset, path, phase='train', shuffle_images=False, transform=None):
        self.phase = phase
        self.shuffle_images = shuffle_images
        self.transform = transform
        self.root_dir = path
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
