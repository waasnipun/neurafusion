import os
from dataset import MiniImageNetDataset
from tools import getDataset
import numpy as np
import matplotlib.pyplot as plt


root_dir = os.path.join(os.getcwd(), 'datasets/miniImageNet')
dataset, label_mapping = getDataset(root_dir, shuffle_images=True)

def visualize(phase, dataset):
    idx = np.random.choice(range(len(dataset)), 5, replace=False)  # randomly pick 5 pictures to show
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('MiniImageNet {} set'.format(phase), fontsize=16)
    for i in range(len(idx)):
        image, label = dataset[idx[i]]
        ax = plt.subplot(1, 5, i + 1)
        plt.tight_layout()
        ax.set_title('class #{}'.format(label_mapping[label]))
        ax.axis('off')
        plt.imshow(np.asarray(image))

    plt.show()

    # print number of images for each class
    print('total number of {} set: {}'.format(phase, len(dataset)))
    for i in label_mapping.keys():
        print('numer of images for class {}: {}'.format(label_mapping[i],
                                                        len([label for _, label in dataset.data if label == i])))



