import os
from torch.utils.data import Dataset
from PIL import Image
import random

class MiniImageNetDataset(Dataset):
    def __init__(self, dataset, path, phase='train', shuffle_images=False, transform=None, start_class=0):
        self.phase = phase
        self.shuffle_images = shuffle_images
        self.transform = transform
        self.root_dir = path

        if phase == 'train':
            self.data = dataset[:int(len(dataset)*0.6)]
        elif phase == 'val':
            self.data = dataset[int(len(dataset)*0.6):int(len(dataset)*0.8)]
        elif phase == 'test':
            self.data = dataset[int(len(dataset)*0.8):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
