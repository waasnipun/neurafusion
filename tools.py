import os
import random

def getDataset(path: str, num_classes: int, shuffle_images=True, num_images_per_class = None) -> tuple:
    images_and_labels = []
    class_mapping = {}
    label_counter = 0
    if num_classes > len(os.listdir(path)):
        raise ValueError(f"num_classes ({num_classes}) cannot be greater than the number of classes in the dataset ({len(os.listdir(path))})")
    
    categories = random.sample(os.listdir(path), k=num_classes)
    for class_folder in categories:
        class_path = os.path.join(path, class_folder)

        if os.path.isdir(class_path):
            class_mapping[label_counter] = class_folder
            images = os.listdir(class_path)

            if num_images_per_class != None:
                num_images_per_class = int(num_images_per_class)

                if num_images_per_class > len(images):
                    raise ValueError(f"num_images_per_class ({num_images_per_class}) cannot be greater than the number of images in the class ({len(images)})")
                images = random.sample(images, k=num_images_per_class)

                print(f"Using {num_images_per_class} images per class {class_folder}")

            label_images = [(os.path.join(class_folder, image), label_counter)
                            for image in images]
            images_and_labels.extend(label_images)
            label_counter += 1

    if shuffle_images:
        random.shuffle(images_and_labels)

    return images_and_labels, class_mapping


# Print class distribution in the datasets
def print_class_distribution(dataset, phase, label_mapping, num_examples=5, chars_per_line=150):
    class_counts = {class_name: 0 for class_name in label_mapping.values()}
    class_examples = {class_name: [] for class_name in label_mapping.values()}

    for _, label in dataset:
        class_name = label_mapping[label]
        class_counts[class_name] += 1

        if len(class_examples[class_name]) < num_examples:
            class_examples[class_name].append(label)

    total_samples = len(dataset)
    num_classes = len(class_counts)

    print(f"Class distribution summary in {phase} set:")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Samples: {total_samples}\n")

    # Print classes in the same line
    current_line_chars = 0
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        class_info = f"{class_name}: {count} samples ({percentage:.2f}%)"
        if current_line_chars + len(class_info) > chars_per_line:
            print()  # Move to the next line
            current_line_chars = 0

        print(f"{class_info} | ",  end="")
        current_line_chars += len(class_info) + 3  # 3 accounts for " | "

    print("\n")  # Ensure the last line ends with a newline
