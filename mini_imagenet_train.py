import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from mini_imagenet_dataset import MiniImageNetDataset
from tools import getDataset, print_class_distribution

from models.resnet18 import ResNet18

batch_size = 64
num_workers = 4
learning_rate = 0.001
num_epochs = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")

root_dir = os.path.join(os.getcwd(), 'datasets/miniImageNet')
dataset, label_mapping = getDataset(root_dir, shuffle_images=True)

transform = transforms.Compose([transforms.Resize((84, 84)),
                                transforms.ToTensor()])

train_dataset = MiniImageNetDataset(dataset=dataset, path=root_dir, phase='train', shuffle_images=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = MiniImageNetDataset(dataset=dataset, path=root_dir, phase='val', shuffle_images=True, transform=transform)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataset = MiniImageNetDataset(dataset=dataset, path=root_dir, phase='test', shuffle_images=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# evaluation function
def eval(net, data_loader):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    net.eval()
    correct = 0.0
    num_images = 0.0
    print('\ntesting/evaluating')
    for i_batch, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outs = net(images)
        _, predicted = torch.max(outs.data, 1)
        correct += (predicted == labels).sum().item()
        num_images += len(labels)
        print('testing/evaluating -> batch: %d correct: %d numb images: %d' % (i_batch, correct, num_images) + '\r', end='')
    acc = correct / num_images
    return acc


# training function
def train(net, train_loader, valid_loader):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    for epoch in range(num_epochs):
        net.train()
        correct = 0.0  # used to accumulate number of correctly recognized images
        num_images = 0.0  # used to accumulate number of images
        total_loss = 0.0

        for i_batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            output_train = net(images)
            loss = criterion(output_train, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicts = output_train.argmax(dim=1)
            correct += predicts.eq(labels).sum().item()
            num_images += len(labels)
            total_loss += loss.item()

            print('training -> epoch: %d, batch: %d, loss: %f' % (epoch, i_batch, loss.item()) + '\r', end='')

        acc = correct / num_images
        acc_eval = eval(net, valid_loader)
        average_loss = total_loss / len(train_loader)
        print('epoch: %d, lr: %f, accuracy: %f, loss: %f, valid accuracy: %f\n' % (epoch, optimizer.param_groups[0]['lr'], acc, average_loss, acc_eval))

        scheduler.step()

    return net

if __name__ == '__main__':
    # Print hyperparameters summary
    print(f"Hyperparameters:")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Number of Workers: {num_workers}\n")
    # print_class_distribution(train_dataset, "Training", label_mapping)
    # print_class_distribution(val_dataset, "Validation", label_mapping)
    # print_class_distribution(test_dataset, "Testing", label_mapping)


    # Instantiate the model
    model = ResNet18(num_classes=100).to(device)

    model = train(net=model, train_loader=train_loader, valid_loader=validation_loader)

    acc_test = eval(model, test_loader)

    print('accuracy on testing data: %f' % acc_test)

    torch.save(model, os.path.join(os.getcwd(), 'pretrained/resnet18_model_full.pth'))