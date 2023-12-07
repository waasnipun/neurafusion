import os
from torch.utils.data import Dataset
from PIL import Image
import random

class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, phase='train', shuffle_images=False, transform=None, start_class=0):
        self.root_dir = os.path.join(root_dir, 'datasets/miniImageNet')
        self.phase = phase
        self.shuffle_images = shuffle_images
        self.transform = transform
        self.start_class = start_class
        self.data, self.class_mapping = self.get_images_and_labels()

    def get_images_and_labels(self):
        images_and_labels = []
        class_mapping = {}
        label_counter = self.start_class

        phase_folder = os.path.join(self.root_dir, self.phase)
        for class_folder in os.listdir(phase_folder):
            class_path = os.path.join(phase_folder, class_folder)
            if os.path.isdir(class_path):
                label_counter += 1
                class_mapping[label_counter] = class_folder
                label_images = [(os.path.join(class_folder, image), label_counter, class_folder)
                                for image in os.listdir(class_path)]
                images_and_labels.extend(label_images)


        if self.shuffle_images:
            random.shuffle(images_and_labels)

        return images_and_labels, class_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, _ = self.data[idx]
        img = Image.open(os.path.join(self.root_dir, self.phase, img_path)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
