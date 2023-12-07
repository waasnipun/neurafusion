import os
from torch.utils.data import Dataset
from PIL import Image
import random

class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, phase='train', shuffle_images=False, transform=None):
        self.root_dir = os.path.join(root_dir, 'datasets/miniImageNet')
        self.folder_path = os.path.join(self.root_dir, phase)
        self.shuffle_images = shuffle_images
        self.transform = transform
        self.data = self.get_images_and_labels()
        self.label_names = list(set([label for _, label in self.data]))
    def get_images_and_labels(self):
        images_and_labels = [(os.path.join(label, image), label)
                             for label in os.listdir(self.folder_path) if
                             os.path.isdir(os.path.join(self.folder_path, label))
                             for image in os.listdir(os.path.join(self.folder_path, label))]
        if self.shuffle_images:
            random.shuffle(images_and_labels)
        return images_and_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(os.path.join(self.folder_path, img_path)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
