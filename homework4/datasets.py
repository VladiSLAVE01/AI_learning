"""
Datasets for MNIST and synthetic CIFAR-10 alternative
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image

# ==================== УПРОЩЕННЫЙ СИНТЕТИЧЕСКИЙ ДАТАСЕТ ====================

class SimpleSyntheticDataset(Dataset):
    """
    Упрощенный синтетический датасет для замены CIFAR-10
    """
    def __init__(self, num_samples=5000, num_classes=10, img_size=32, channels=3, transform=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels
        self.transform = transform

        # Генерируем данные
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            # Создаем случайное изображение
            img = np.random.randn(channels, img_size, img_size) * 0.5

            # Добавляем простые паттерны
            if random.random() > 0.5:
                # Добавляем цветной прямоугольник
                x = random.randint(5, img_size - 5)
                y = random.randint(5, img_size - 5)
                w = random.randint(3, 8)
                h = random.randint(3, 8)
                color = np.random.rand(channels)
                for c in range(channels):
                    img[c, y:y+h, x:x+w] = color[c]

            if random.random() > 0.6:
                # Добавляем шум
                noise = np.random.randn(channels, img_size, img_size) * 0.1
                img = img + noise

            # Нормализуем
            img = np.clip(img, -1, 1)

            # Преобразуем в формат HWC (height, width, channels) для PIL
            img_hwc = np.transpose(img, (1, 2, 0))
            # Масштабируем к [0, 255] для PIL
            img_hwc = ((img_hwc + 1) / 2 * 255).astype(np.uint8)

            # Создаем PIL Image
            img_pil = Image.fromarray(img_hwc)

            self.data.append(img_pil)
            self.labels.append(random.randint(0, num_classes - 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

# ==================== MNIST ДАТАСЕТ ====================

class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

# ==================== ФУНКЦИИ ЗАГРУЗКИ ====================

def get_mnist_loaders(batch_size=64):
    """Загрузка MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_cifar_loaders(batch_size=64):
    """
    Загрузка синтетического датасета (замена CIFAR-10)
    """
    # Трансформации как для CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Создаем синтетические данные
    train_dataset = SimpleSyntheticDataset(
        num_samples=3000,  # Меньше для скорости
        num_classes=10,
        transform=transform_train
    )
    test_dataset = SimpleSyntheticDataset(
        num_samples=500,
        num_classes=10,
        transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader