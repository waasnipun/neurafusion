import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from mini_imagenet_dataset import MiniImageNetDataset

from models.resnet18 import ResNet18

batch_size = 64
num_workers = 4
learning_rate = 0.001
num_epochs = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")

transform = transforms.Compose([transforms.Resize((84, 84)),
                                transforms.ToTensor()])

train_dataset = MiniImageNetDataset(root_dir=os.getcwd(), phase='train', shuffle_images=True, transform=transform, start_class=0)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = MiniImageNetDataset(root_dir=os.getcwd(), phase='val', shuffle_images=True, transform=transform, start_class=64)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dataset = MiniImageNetDataset(root_dir=os.getcwd(), phase='test', shuffle_images=True, transform=transform, start_class=80)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == '__main__':
    # Instantiate the model
    model = ResNet18(num_classes=100).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0.0
        num_images = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f'Epoch: {epoch} Loss: {loss.item()}', end='\r')

            predicts = outputs.argmax(dim=1)
            correct += predicts.eq(labels).sum().item()
            num_images += len(labels)

        acc = correct / num_images
        average_loss = total_loss / len(train_loader)
        print('epoch: %d, lr: %f, accuracy: %f, average loss: %f, valid accuracy: %f' % (epoch, optimizer.param_groups[0]['lr'], acc, average_loss, 0.0))


    # Validation loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in validation_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f'Predicted: {predicted} | Actual: {labels}')

    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy * 100}%')

    # Test loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100}%')
