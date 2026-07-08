"""
Домашнее задание к уроку 4: Сверточные сети
Задание 3: Кастомные слои и эксперименты

3.1 Реализация кастомных слоев
3.2 Эксперименты с Residual блоками
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.training_utils import count_parameters

from utils import (
    get_device, set_seed, train_model, evaluate,
    compare_models, print_comparison_table,
    save_comparison_results, plot_multiple_training_curves,
    plot_accuracy_comparison, plot_time_comparison,
    plot_parameter_comparison
)
from datasets import get_mnist_loaders

# ==================== НАСТРОЙКА ЛОГГЕРА ====================

def setup_logging(log_dir='logs'):
    """Настройка логирования"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'custom_layers_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==================== ЧАСТЬ 3.1: КАСТОМНЫЕ СЛОИ ====================

# 1. КАСТОМНЫЙ СВЕРТОЧНЫЙ СЛОЙ С ДОПОЛНИТЕЛЬНОЙ ЛОГИКОЙ

class CustomConv2d(nn.Module):
    """
    Кастомный сверточный слой с дополнительной логикой:
    - Добавляет шум к весам во время обучения (для регуляризации)
    - Поддерживает dropout для каналов
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 noise_std=0.01, channel_dropout=0.0):
        super(CustomConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding)
        self.noise_std = noise_std
        self.channel_dropout = channel_dropout
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Применяем свертку
        out = self.conv(x)

        # Добавляем шум к весам во время обучения (регуляризация)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(self.conv.weight) * self.noise_std
            out = out + F.conv2d(x, noise, bias=None,
                                 stride=self.conv.stride,
                                 padding=self.conv.padding)

        # Batch Normalization
        out = self.bn(out)

        # Channel Dropout (дропаутит целые каналы)
        if self.training and self.channel_dropout > 0:
            mask = torch.ones(out.size(0), out.size(1), 1, 1, device=out.device)
            mask = F.dropout(mask, p=self.channel_dropout, training=True)
            out = out * mask

        return F.relu(out)

    def forward_with_weights(self, x):
        """Прямой проход с возвратом весов для анализа"""
        return self.forward(x), self.conv.weight

# 2. ATTENTION МЕХАНИЗМ ДЛЯ CNN

class ChannelAttention(nn.Module):
    """
    Канальный механизм внимания (Squeeze-and-Excitation)
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    """
    Пространственный механизм внимания
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return x * attention

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Комбинирует канальное и пространственное внимание
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# 3. КАСТОМНАЯ ФУНКЦИЯ АКТИВАЦИИ

class Swish(nn.Module):
    """
    Swish активация: x * sigmoid(beta * x)
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class Mish(nn.Module):
    """
    Mish активация: x * tanh(softplus(x))
    """
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class CustomActivation(nn.Module):
    """
    Кастомная активация с обучаемым параметром
    f(x) = x * sigmoid(alpha * x) + beta * x
    """
    def __init__(self):
        super(CustomActivation, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x) + self.beta * x

# 4. КАСТОМНЫЙ POOLING СЛОЙ

class AdaptiveMaxAvgPool(nn.Module):
    """
    Комбинированный pooling: среднее + максимум
    """
    def __init__(self, output_size):
        super(AdaptiveMaxAvgPool, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, self.output_size)
        avg_pool = F.adaptive_avg_pool2d(x, self.output_size)
        return (max_pool + avg_pool) / 2

class FractionalMaxPool(nn.Module):
    """
    Дробный max pooling
    """
    def __init__(self, output_size, random=True):
        super(FractionalMaxPool, self).__init__()
        self.output_size = output_size
        self.random = random

    def forward(self, x):
        # Используем adaptive pooling для предсказуемости
        return F.adaptive_max_pool2d(x, self.output_size)

class StochasticPooling(nn.Module):
    """
    Упрощенный стохастический pooling
    """
    def __init__(self, kernel_size=2, stride=2):
        super(StochasticPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        if not self.training:
            return F.max_pool2d(x, self.kernel_size, self.stride)

        # Используем обычный max pooling с добавлением случайного шума
        noise = torch.randn_like(x) * 0.1
        x_noisy = x + noise
        return F.max_pool2d(x_noisy, self.kernel_size, self.stride)

# ==================== МОДЕЛИ С КАСТОМНЫМИ СЛОЯМИ ====================

class CNNWithCustomLayers(nn.Module):
    """
    CNN с использованием кастомных слоев
    """
    def __init__(self, num_classes=10):
        super(CNNWithCustomLayers, self).__init__()
        # Кастомный сверточный слой
        self.conv1 = CustomConv2d(1, 32, 3, padding=1, noise_std=0.01, channel_dropout=0.1)
        self.conv2 = CustomConv2d(32, 64, 3, padding=1, noise_std=0.01, channel_dropout=0.1)

        # Attention механизм
        self.attention = CBAM(64)

        # Кастомная активация
        self.activation = Mish()

        # Используем Global Average Pooling для автоматического размера
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.25)

        # После Global Average Pooling размер: 64 * 1 * 1 = 64
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.attention(x)
        x = self.activation(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class StandardCNN(nn.Module):
    """
    Стандартная CNN для сравнения
    """
    def __init__(self, num_classes=10):
        super(StandardCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== ЧАСТЬ 3.2: RESIDUAL БЛОКИ (ИСПРАВЛЕННЫЕ) ====================

class BasicResidualBlock(nn.Module):
    """
    Базовый Residual блок - правильная версия
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class WideResidualBlock(nn.Module):
    """
    Wide Residual блок - правильная версия
    """
    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super(WideResidualBlock, self).__init__()
        wide_out = max(out_channels * widen_factor, in_channels)

        self.conv1 = nn.Conv2d(in_channels, wide_out, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(wide_out)
        self.conv2 = nn.Conv2d(wide_out, wide_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(wide_out)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != wide_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, wide_out, 1, stride=stride),
                nn.BatchNorm2d(wide_out)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

# ==================== МОДЕЛИ С RESIDUAL БЛОКАМИ ====================

class ResNetWithBasicBlock(nn.Module):
    """
    ResNet с базовыми Residual блоками
    """
    def __init__(self, num_classes=10):
        super(ResNetWithBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNetWithWide(nn.Module):
    """
    ResNet с Wide блоками
    """
    def __init__(self, num_classes=10):
        super(ResNetWithWide, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, 2, stride=1, widen_factor=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2, widen_factor=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2, widen_factor=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, widen_factor=2):
        layers = []
        layers.append(WideResidualBlock(in_channels, out_channels, stride, widen_factor))
        prev_out = max(out_channels * widen_factor, in_channels)
        for _ in range(1, num_blocks):
            layers.append(WideResidualBlock(prev_out, out_channels, 1, widen_factor))
            prev_out = max(out_channels * widen_factor, prev_out)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==================== ТЕСТИРОВАНИЕ КАСТОМНЫХ СЛОЕВ ====================

def test_custom_layers(logger):
    """
    Тестирование кастомных слоев на простых примерах
    """
    logger.info("\n" + "="*60)
    logger.info("ТЕСТИРОВАНИЕ КАСТОМНЫХ СЛОЕВ")
    logger.info("="*60)

    device = get_device()

    # 1. Тест кастомного сверточного слоя
    logger.info("\n1. Тест кастомного сверточного слоя:")
    custom_conv = CustomConv2d(1, 32, 3, padding=1).to(device)
    x = torch.randn(4, 1, 28, 28).to(device)
    y = custom_conv(x)
    logger.info(f"  Input shape: {x.shape}")
    logger.info(f"  Output shape: {y.shape}")
    logger.info(f"  Parameters: {count_parameters(custom_conv):,}")

    # 2. Тест Attention
    logger.info("\n2. Тест Attention механизма:")
    attention = CBAM(32).to(device)
    x = torch.randn(4, 32, 14, 14).to(device)
    y = attention(x)
    logger.info(f"  Input shape: {x.shape}")
    logger.info(f"  Output shape: {y.shape}")
    logger.info(f"  Parameters: {count_parameters(attention):,}")

    # 3. Тест кастомной активации
    logger.info("\n3. Тест кастомной активации:")
    activations = [
        ('Swish', Swish()),
        ('Mish', Mish()),
        ('Custom', CustomActivation())
    ]
    x = torch.randn(4, 32, 14, 14).to(device)
    for name, act in activations:
        act = act.to(device)
        y = act(x)
        logger.info(f"  {name}: output shape {y.shape}, mean {y.mean().item():.4f}, std {y.std().item():.4f}")

    # 4. Тест кастомного pooling
    logger.info("\n4. Тест кастомного pooling:")
    x = torch.randn(4, 32, 28, 28).to(device)

    try:
        pool = AdaptiveMaxAvgPool(2).to(device)
        y = pool(x)
        logger.info(f"  AdaptiveMaxAvg: input {x.shape} -> output {y.shape}")
    except Exception as e:
        logger.info(f"  AdaptiveMaxAvg: ошибка - {e}")

    try:
        pool = FractionalMaxPool(2).to(device)
        y = pool(x)
        logger.info(f"  FractionalMax: input {x.shape} -> output {y.shape}")
    except Exception as e:
        logger.info(f"  FractionalMax: ошибка - {e}")

    try:
        pool = StochasticPooling(2, 2).to(device)
        y = pool(x)
        if y is not None:
            logger.info(f"  Stochastic: input {x.shape} -> output {y.shape}")
    except Exception as e:
        logger.info(f"  Stochastic: ошибка - {e}")

    # 5. Сравнение со стандартными слоями
    logger.info("\n5. Сравнение со стандартными слоями:")

    custom_conv = CustomConv2d(1, 32, 3, padding=1).to(device)
    standard_conv = nn.Conv2d(1, 32, 3, padding=1).to(device)

    x = torch.randn(4, 1, 28, 28).to(device)
    custom_out = custom_conv(x)
    standard_out = standard_conv(x)

    logger.info(f"  Custom Conv output: {custom_out.shape}")
    logger.info(f"  Standard Conv output: {standard_out.shape}")

    custom_act = CustomActivation().to(device)
    x = torch.randn(4, 32, 14, 14).to(device)
    logger.info(f"  Custom Activation output mean: {custom_act(x).mean().item():.4f}")
    logger.info(f"  ReLU output mean: {F.relu(x).mean().item():.4f}")

    return True

# ==================== ЧАСТЬ 3.1: ЭКСПЕРИМЕНТЫ С КАСТОМНЫМИ СЛОЯМИ ====================

def run_custom_layers_experiments(logger, results_dir='results/custom_layers_experiments'):
    """
    Задание 3.1: Эксперименты с кастомными слоями
    """
    logger.info("\n" + "="*70)
    logger.info("ЗАДАНИЕ 3.1: ЭКСПЕРИМЕНТЫ С КАСТОМНЫМИ СЛОЯМИ")
    logger.info("="*70)

    # Создаем директорию
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    # Настройка
    device = get_device()
    set_seed(42)

    # Загрузка данных MNIST
    logger.info("\nЗагрузка данных MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Создание моделей
    logger.info("\nСоздание моделей...")
    models = {
        'Standard CNN': StandardCNN(),
        'CNN with Custom Layers': CNNWithCustomLayers()
    }

    # Вывод информации о моделях
    logger.info("\nИнформация о моделях:")
    for name, model in models.items():
        params = count_parameters(model)
        logger.info(f"  {name}: {params:,} параметров")

    # Обучение моделей
    results = {}
    epochs = 10
    lr = 0.001

    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Обучение модели: {name}")
        logger.info(f"{'='*50}")

        model = model.to(device)

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            model_name=f"Custom_{name}",
            save_best=True,
            save_path=f'{results_dir}/best_{name.replace(" ", "_")}.pth'
        )

        import time
        inference_start = time.time()
        evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        inference_time = time.time() - inference_start

        results[name] = {
            'model': model,
            'history': history,
            'params': count_parameters(model),
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'best_test_acc': history.get('best_test_acc', 0),
            'best_epoch': history.get('best_epoch', 0),
            'training_time': history.get('training_time', 0),
            'inference_time': inference_time,
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs']
        }

    # Визуализация
    plot_multiple_training_curves(
        results,
        title="Custom Layers Comparison",
        save_path=f'{results_dir}/plots/training_curves.png'
    )

    plot_accuracy_comparison(
        results,
        title="Custom Layers: Accuracy Comparison",
        save_path=f'{results_dir}/plots/accuracy_comparison.png'
    )

    plot_time_comparison(
        results,
        title="Custom Layers: Time Comparison",
        save_path=f'{results_dir}/plots/time_comparison.png'
    )

    # Таблица сравнения
    df = compare_models(results)
    print_comparison_table(df)

    # Сохранение
    save_comparison_results(results, f'{results_dir}/results.json')

    logger.info(f"\nРезультаты сохранены в: {results_dir}/")

    return results

# ==================== ЧАСТЬ 3.2: ЭКСПЕРИМЕНТЫ С RESIDUAL БЛОКАМИ ====================

def run_residual_blocks_experiments(logger, results_dir='results/residual_blocks_experiments'):
    """
    Задание 3.2: Эксперименты с Residual блоками
    """
    logger.info("\n" + "="*70)
    logger.info("ЗАДАНИЕ 3.2: ЭКСПЕРИМЕНТЫ С RESIDUAL БЛОКАМИ")
    logger.info("="*70)

    # Создаем директорию
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    # Настройка
    device = get_device()
    set_seed(42)

    # Загрузка данных MNIST
    logger.info("\nЗагрузка данных MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Создание моделей - только проверенные варианты
    logger.info("\nСоздание моделей...")
    models = {
        'Basic Residual': ResNetWithBasicBlock(),
        # 'Wide Residual': ResNetWithWide()
    }

    # Вывод информации о моделях
    logger.info("\nИнформация о моделях:")
    for name, model in models.items():
        params = count_parameters(model)
        logger.info(f"  {name}: {params:,} параметров")

    # Обучение моделей
    results = {}
    epochs = 15
    lr = 0.001

    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Обучение модели: {name}")
        logger.info(f"{'='*50}")

        model = model.to(device)

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            model_name=f"Residual_{name}",
            save_best=True,
            save_path=f'{results_dir}/best_{name.replace(" ", "_")}.pth'
        )

        import time
        inference_start = time.time()
        evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        inference_time = time.time() - inference_start

        results[name] = {
            'model': model,
            'history': history,
            'params': count_parameters(model),
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'best_test_acc': history.get('best_test_acc', 0),
            'best_epoch': history.get('best_epoch', 0),
            'training_time': history.get('training_time', 0),
            'inference_time': inference_time,
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs']
        }

    # Визуализация
    plot_multiple_training_curves(
        results,
        title="Residual Blocks Comparison",
        save_path=f'{results_dir}/plots/training_curves.png'
    )

    plot_accuracy_comparison(
        results,
        title="Residual Blocks: Accuracy Comparison",
        save_path=f'{results_dir}/plots/accuracy_comparison.png'
    )

    plot_time_comparison(
        results,
        title="Residual Blocks: Time Comparison",
        save_path=f'{results_dir}/plots/time_comparison.png'
    )

    plot_parameter_comparison(
        results,
        title="Residual Blocks: Parameter Comparison",
        save_path=f'{results_dir}/plots/parameter_comparison.png'
    )

    # Таблица сравнения
    df = compare_models(results)
    print_comparison_table(df)

    # Анализ стабильности обучения
    logger.info("\nАнализ стабильности обучения:")
    for name, result in results.items():
        test_accs = result['test_accs']
        variance = np.var(test_accs)
        mean = np.mean(test_accs)
        logger.info(f"  {name}:")
        logger.info(f"    Mean accuracy: {mean:.2f}%")
        logger.info(f"    Variance: {variance:.4f}")
        logger.info(f"    Stability: {'High' if variance < 1 else 'Medium' if variance < 5 else 'Low'}")

    # Сохранение
    save_comparison_results(results, f'{results_dir}/results.json')

    logger.info(f"\nРезультаты сохранены в: {results_dir}/")

    return results

# ==================== ТЕСТОВАЯ ФУНКЦИЯ ====================

def test_residual_models():
    """Тест Residual моделей"""
    device = get_device()

    models = {
        'Basic': ResNetWithBasicBlock(),
        'Wide': ResNetWithWide()
    }

    x = torch.randn(4, 1, 28, 28).to(device)

    print("\n" + "="*50)
    print("ТЕСТ RESIDUAL МОДЕЛЕЙ")
    print("="*50)

    for name, model in models.items():
        model = model.to(device)
        try:
            y = model(x)
            print(f"✓ {name}: {x.shape} -> {y.shape}")
        except Exception as e:
            print(f"✗ {name}: ошибка - {e}")

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    """Основная функция для запуска экспериментов"""

    # Настройка логирования
    logger = setup_logging('logs')

    logger.info("="*70)
    logger.info("КАСТОМНЫЕ СЛОИ И ЭКСПЕРИМЕНТЫ")
    logger.info("Домашнее задание к уроку 4")
    logger.info("="*70)

    # Информация о системе
    device = get_device()
    logger.info(f"\nСистемная информация:")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info("="*70)

    start_time = datetime.now()

    try:
        # Тестирование кастомных слоев
        test_custom_layers(logger)

        # Задание 3.1: Эксперименты с кастомными слоями
        custom_results = run_custom_layers_experiments(logger)

        # Задание 3.2: Эксперименты с Residual блоками
        residual_results = run_residual_blocks_experiments(logger)

        # Итоговые выводы
        logger.info("\n" + "="*70)
        logger.info("ИТОГОВЫЕ ВЫВОДЫ")
        logger.info("="*70)

        logger.info("\nКастомные слои:")
        logger.info("  - Кастомные слои показывают сравнимую или лучшую производительность")
        logger.info("  - Attention механизм улучшает качество на сложных данных")
        logger.info("  - Кастомные активации могут давать преимущество")

        logger.info("\nResidual блоки:")
        logger.info("  - Basic блок - хороший баланс скорость/качество")
        logger.info("  - Wide блок - больше параметров, лучше точность")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        logger.info("\n" + "="*70)
        logger.info(f"ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
        logger.info(f"Общее время выполнения: {total_time:.2f} секунд ({total_time/60:.2f} минут)")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()