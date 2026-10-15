"""Part 2 — Train the Part 1 CNN on CIFAR10.

CIFAR10 images are 32x32; they are resized to 64x64 so the Part 1
architecture can be reused unchanged. Trains for 5 epochs on CPU,
then saves the weights to app/cifar10_cnn.pt for the /classify API.
"""

import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, 'app')
from cnn_model import CNNClassifier

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001


def main():
    device = torch.device('cpu')
    print('device:', device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # match the Part 1 input size
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.ImageFolder(root='./data/cifar10/train', transform=transform)
    test_set = datasets.ImageFolder(root='./data/cifar10/test', transform=transform)
    # ImageFolder sorts classes alphabetically, which matches the
    # torchvision CIFAR10 class order: airplane..truck
    assert train_set.classes == ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                 'dog', 'frog', 'horse', 'ship', 'truck']
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=2)
    print('train images:', len(train_set), '| test images:', len(test_set))

    model = CNNClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        print('epoch %d/%d - loss: %.4f - train acc: %.2f%% - %.0fs'
              % (epoch + 1, EPOCHS, running_loss / total, 100 * correct / total, time.time() - t0))

    # test accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels.to(device)).sum().item()
            total += labels.size(0)
    print('test accuracy: %.2f%% (%d/%d)' % (100 * correct / total, correct, total))

    torch.save(model.state_dict(), 'app/cifar10_cnn.pt')
    print('saved app/cifar10_cnn.pt')


if __name__ == '__main__':
    main()
