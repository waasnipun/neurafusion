import os
import random
def getDataset(path: str, shuffle_images=False) -> tuple:
    images_and_labels = []
    class_mapping = {}
    label_counter = 0
    for class_folder in os.listdir(path):
        class_path = os.path.join(path, class_folder)
        if os.path.isdir(class_path):
            label_counter += 1
            class_mapping[label_counter] = class_folder
            label_images = [(os.path.join(class_folder, image), label_counter)
                            for image in os.listdir(class_path)]
            images_and_labels.extend(label_images)

    if shuffle_images:
        random.shuffle(images_and_labels)

    return images_and_labels, class_mapping