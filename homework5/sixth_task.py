import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from augmentations_basics.datasets import CustomImageDataset
from torchvision import transforms, models
from torch.utils.data import DataLoader


def plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_epoch(model, data_loader, loss_fn, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    model.train()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.size(0)

    return total_loss / len(data_loader), correct / total


def main():
    train_dataset = CustomImageDataset('data/train', transform=transforms.ToTensor())
    test_dataset = CustomImageDataset('data/test', transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.efficientnet_b0(weights='IMAGENET1K_V1', progress=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.get_class_names()))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(10):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, is_test=True)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    plot_metrics(train_losses, train_accuracies, test_losses, test_accuracies)


if __name__ == '__main__':
    main()
