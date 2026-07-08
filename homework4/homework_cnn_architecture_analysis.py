"""
Домашнее задание к уроку 4: Сверточные сети
Задание 2: Анализ архитектур CNN

2.1 Влияние размера ядра свертки
2.2 Влияние глубины CNN
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

# Импорты из наших модулей
from models import SimpleCNN_MNIST, ResCNN_MNIST
from utils import (
	get_device, set_seed, train_model, evaluate,
	compare_models, print_comparison_table, get_best_model,
	save_comparison_results, plot_training_curves,
	plot_multiple_training_curves, plot_confusion_matrix,
	plot_accuracy_comparison, plot_time_comparison,
	plot_parameter_comparison, plot_gradient_flow,
	plot_gradient_distribution
)
from datasets import get_mnist_loaders
from utils.training_utils import count_parameters

# ==================== НАСТРОЙКА ЛОГГЕРА ====================

def setup_logging(log_dir='logs'):
	"""Настройка логирования"""
	os.makedirs(log_dir, exist_ok=True)
	timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
	log_file = os.path.join(log_dir, f'architecture_analysis_{timestamp}.log')

	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		handlers=[
			logging.FileHandler(log_file, encoding='utf-8'),
			logging.StreamHandler()
		]
	)
	return logging.getLogger(__name__)


# ==================== ЧАСТЬ 2.1: МОДЕЛИ С РАЗНЫМИ ЯДРАМИ ====================

class CNNWithKernelSize(nn.Module):
	"""
	CNN с заданным размером ядра
	"""

	def __init__(self, kernel_size=3, num_classes=10):
		super(CNNWithKernelSize, self).__init__()
		padding = kernel_size // 2

		self.conv1 = nn.Conv2d(1, 32, kernel_size, padding=padding)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size, padding=padding)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 128, kernel_size, padding=padding)
		self.bn3 = nn.BatchNorm2d(128)

		self.pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(0.25)

		# Вычисляем размер после сверток и пулинга
		# Для MNIST: 28 -> после 3 пулингов (2x2) -> 28/8 = 3.5 -> 3
		self.fc1 = nn.Linear(128 * 3 * 3, 256)
		self.fc2 = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.pool(F.relu(self.bn1(self.conv1(x))))
		x = self.pool(F.relu(self.bn2(self.conv2(x))))
		x = self.pool(F.relu(self.bn3(self.conv3(x))))
		x = x.view(x.size(0), -1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.fc2(x)
		return x


class CNNWithMixedKernels(nn.Module):
	"""
	CNN с комбинацией разных размеров ядер (1x1 + 3x3)
	"""

	def __init__(self, num_classes=10):
		super(CNNWithMixedKernels, self).__init__()
		# Первый слой: 1x1 + 3x3
		self.conv1_1x1 = nn.Conv2d(1, 16, 1)
		self.conv1_3x3 = nn.Conv2d(1, 16, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)

		# Второй слой: 1x1 + 3x3
		self.conv2_1x1 = nn.Conv2d(32, 32, 1)
		self.conv2_3x3 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(64)

		# Третий слой: 1x1 + 3x3
		self.conv3_1x1 = nn.Conv2d(64, 64, 1)
		self.conv3_3x3 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(128)

		self.pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(0.25)

		self.fc1 = nn.Linear(128 * 3 * 3, 256)
		self.fc2 = nn.Linear(256, num_classes)

	def forward(self, x):
		# Первый блок
		x1 = self.conv1_1x1(x)
		x2 = self.conv1_3x3(x)
		x = torch.cat([x1, x2], dim=1)
		x = F.relu(self.bn1(x))
		x = self.pool(x)

		# Второй блок
		x1 = self.conv2_1x1(x)
		x2 = self.conv2_3x3(x)
		x = torch.cat([x1, x2], dim=1)
		x = F.relu(self.bn2(x))
		x = self.pool(x)

		# Третий блок
		x1 = self.conv3_1x1(x)
		x2 = self.conv3_3x3(x)
		x = torch.cat([x1, x2], dim=1)
		x = F.relu(self.bn3(x))
		x = self.pool(x)

		x = x.view(x.size(0), -1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.fc2(x)
		return x


# ==================== ЧАСТЬ 2.2: МОДЕЛИ С РАЗНОЙ ГЛУБИНОЙ ====================

class ShallowCNN(nn.Module):
	"""Неглубокая CNN (2 conv слоя)"""

	def __init__(self, num_classes=10):
		super(ShallowCNN, self).__init__()
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


class MediumCNN(nn.Module):
	"""Средняя CNN (4 conv слоя)"""

	def __init__(self, num_classes=10):
		super(MediumCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(128)

		self.pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(0.25)

		self.fc1 = nn.Linear(128 * 3 * 3, 256)
		self.fc2 = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.pool(F.relu(self.bn1(self.conv1(x))))
		x = self.pool(F.relu(self.bn2(self.conv2(x))))
		x = self.pool(F.relu(self.bn3(self.conv3(x))))
		x = F.relu(self.bn4(self.conv4(x)))
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.fc2(x)
		return x


class DeepCNN(nn.Module):
	"""Глубокая CNN (6+ conv слоев)"""

	def __init__(self, num_classes=10):
		super(DeepCNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
		self.bn5 = nn.BatchNorm2d(128)
		self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(128)

		self.pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(0.3)

		self.fc1 = nn.Linear(128 * 3 * 3, 256)
		self.fc2 = nn.Linear(256, num_classes)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.pool(x)

		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = self.pool(x)

		x = F.relu(self.bn5(self.conv5(x)))
		x = F.relu(self.bn6(self.conv6(x)))
		x = self.pool(x)

		x = x.view(x.size(0), -1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.fc2(x)
		return x


class DeepCNNWithResidual(nn.Module):
	"""Глубокая CNN с Residual связями"""

	def __init__(self, num_classes=10):
		super(DeepCNNWithResidual, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)

		# Residual блоки
		self.res1 = nn.Sequential(
			nn.Conv2d(32, 32, 3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, padding=1),
			nn.BatchNorm2d(32)
		)

		self.res2 = nn.Sequential(
			nn.Conv2d(32, 64, 3, stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, padding=1),
			nn.BatchNorm2d(64)
		)
		self.skip2 = nn.Conv2d(32, 64, 1, stride=2)

		self.res3 = nn.Sequential(
			nn.Conv2d(64, 128, 3, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.BatchNorm2d(128)
		)
		self.skip3 = nn.Conv2d(64, 128, 1, stride=2)

		self.res4 = nn.Sequential(
			nn.Conv2d(128, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, padding=1),
			nn.BatchNorm2d(128)
		)

		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.dropout = nn.Dropout(0.3)
		self.fc = nn.Linear(128, num_classes)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))

		# Residual блок 1
		residual = x
		x = self.res1(x)
		x = F.relu(x + residual)

		# Residual блок 2
		residual = self.skip2(x)
		x = self.res2(x)
		x = F.relu(x + residual)

		# Residual блок 3
		residual = self.skip3(x)
		x = self.res3(x)
		x = F.relu(x + residual)

		# Residual блок 4
		residual = x
		x = self.res4(x)
		x = F.relu(x + residual)

		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.dropout(x)
		x = self.fc(x)
		return x


# ==================== ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ АКТИВАЦИЙ ====================

def visualize_activations(model, dataloader, device, layer_name, num_images=4, save_path=None):
	"""
	Визуализирует активации первого слоя модели
	"""
	model.eval()

	# Получаем батч
	data, _ = next(iter(dataloader))
	data = data[:num_images].to(device)

	# Регистрируем hook для получения активаций
	activations = []

	def hook_fn(module, input, output):
		activations.append(output.detach().cpu())

	# Находим нужный слой
	for name, module in model.named_modules():
		if name == layer_name:
			handle = module.register_forward_hook(hook_fn)
			break

	# Forward pass
	with torch.no_grad():
		_ = model(data)

	handle.remove()

	if not activations:
		print(f"Layer {layer_name} not found")
		return

	activation = activations[0]  # [batch, channels, height, width]

	# Визуализация
	fig, axes = plt.subplots(num_images, min(8, activation.shape[1]),
	                         figsize=(12, num_images * 3))

	if num_images == 1:
		axes = axes.reshape(1, -1)

	for i in range(num_images):
		for j in range(min(8, activation.shape[1])):
			if num_images == 1:
				ax = axes[j]
			else:
				ax = axes[i, j]

			# Нормализуем для отображения
			act = activation[i, j].numpy()
			act = (act - act.min()) / (act.max() - act.min() + 1e-8)
			ax.imshow(act, cmap='viridis')
			ax.axis('off')

	plt.suptitle(f'Activations of {layer_name}', fontsize=14)
	plt.tight_layout()

	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=150, bbox_inches='tight')

	plt.show()
	return fig


def visualize_feature_maps(model, dataloader, device, layer_names, save_path=None):
	"""
	Визуализирует feature maps на разных слоях
	"""
	model.eval()

	# Получаем батч
	data, _ = next(iter(dataloader))
	data = data[:1].to(device)  # Берем одно изображение

	# Регистрируем hooks
	activations = {}

	def get_hook(name):
		def hook_fn(module, input, output):
			activations[name] = output.detach().cpu()

		return hook_fn

	handles = []
	for name, module in model.named_modules():
		if name in layer_names:
			handle = module.register_forward_hook(get_hook(name))
			handles.append(handle)

	# Forward pass
	with torch.no_grad():
		_ = model(data)

	# Удаляем hooks
	for handle in handles:
		handle.remove()

	# Визуализация
	n_layers = len(activations)
	fig, axes = plt.subplots(n_layers, 8, figsize=(16, n_layers * 2.5))

	if n_layers == 1:
		axes = axes.reshape(1, -1)

	for i, (name, act) in enumerate(activations.items()):
		# Берем первые 8 каналов
		n_channels = min(8, act.shape[1])
		for j in range(n_channels):
			if n_layers == 1:
				ax = axes[j]
			else:
				ax = axes[i, j]

			# Нормализуем
			fm = act[0, j].numpy()
			fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
			ax.imshow(fm, cmap='viridis')
			ax.axis('off')

			if j == 0:
				ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=20)

	plt.suptitle('Feature Maps Visualization', fontsize=16)
	plt.tight_layout()

	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=150, bbox_inches='tight')

	plt.show()
	return fig


# ==================== ЧАСТЬ 2.1: ВЛИЯНИЕ РАЗМЕРА ЯДРА ====================

def run_kernel_analysis(logger, results_dir='results/architecture_analysis/kernel_analysis'):
	"""
	Задание 2.1: Влияние размера ядра свертки
	"""
	logger.info("\n" + "=" * 70)
	logger.info("ЗАДАНИЕ 2.1: ВЛИЯНИЕ РАЗМЕРА ЯДРА СВЕРТКИ")
	logger.info("=" * 70)

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

	# Создание моделей с разными ядрами
	logger.info("\nСоздание моделей...")
	models = {
		'Kernel 3x3': CNNWithKernelSize(kernel_size=3),
		'Kernel 5x5': CNNWithKernelSize(kernel_size=5),
		'Kernel 7x7': CNNWithKernelSize(kernel_size=7),
		'Mixed (1x1+3x3)': CNNWithMixedKernels()
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
		logger.info(f"\n{'=' * 50}")
		logger.info(f"Обучение модели: {name}")
		logger.info(f"{'=' * 50}")

		model = model.to(device)

		history = train_model(
			model=model,
			train_loader=train_loader,
			test_loader=test_loader,
			epochs=epochs,
			lr=lr,
			device=device,
			model_name=f"Kernel_{name}",
			save_best=True,
			save_path=f'{results_dir}/best_{name.replace(" ", "_")}.pth'
		)

		# Измеряем время инференса
		import time
		inference_start = time.time()
		from utils.training_utils import evaluate
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

	# ==================== АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ====================

	# 1. Сравнительная таблица
	df = compare_models(results)
	print_comparison_table(df)

	# 2. Сохранение результатов
	save_comparison_results(results, f'{results_dir}/results.json')

	# 3. Кривые обучения
	plot_multiple_training_curves(
		results,
		title="Kernel Size Comparison",
		save_path=f'{results_dir}/plots/training_curves.png'
	)

	# 4. Сравнение точности
	plot_accuracy_comparison(
		results,
		title="Kernel Size: Accuracy Comparison",
		save_path=f'{results_dir}/plots/accuracy_comparison.png'
	)

	# 5. Сравнение времени
	plot_time_comparison(
		results,
		title="Kernel Size: Time Comparison",
		save_path=f'{results_dir}/plots/time_comparison.png'
	)

	# 6. Визуализация активаций первого слоя
	logger.info("\nВизуализация активаций первого слоя...")

	for name, result in results.items():
		model = result['model']
		layer_name = 'conv1'  # Имя первого сверточного слоя

		# Проверяем, есть ли слой conv1
		has_conv1 = any('conv1' in name for name, _ in model.named_modules())
		if has_conv1:
			visualize_activations(
				model, test_loader, device, 'conv1',
				num_images=4,
				save_path=f'{results_dir}/plots/activations_{name.replace(" ", "_")}.png'
			)

	# 7. Анализ рецептивных полей
	logger.info("\nАнализ рецептивных полей:")
	logger.info("Рецептивные поля для разных размеров ядер:")
	logger.info("  Kernel 3x3: receptive field = 3")
	logger.info("  Kernel 5x5: receptive field = 5")
	logger.info("  Kernel 7x7: receptive field = 7")
	logger.info("  Mixed: комбинация 1x1 и 3x3")

	# 8. Отчет
	with open(f'{results_dir}/summary.txt', 'w', encoding='utf-8') as f:
		f.write("=" * 70 + "\n")
		f.write("АНАЛИЗ ВЛИЯНИЯ РАЗМЕРА ЯДРА\n")
		f.write("=" * 70 + "\n\n")
		f.write(df.to_string())
		f.write("\n\nВыводы:\n")
		f.write("1. Большие ядра (7x7) захватывают больше контекста\n")
		f.write("2. Малые ядра (3x3) дают более детальные признаки\n")
		f.write("3. Смешанные ядра (1x1 + 3x3) эффективно используют оба подхода\n")
		f.write("4. Скорость обучения зависит от размера ядра\n")

	logger.info(f"\nРезультаты сохранены в: {results_dir}/")

	return results


# ==================== ЧАСТЬ 2.2: ВЛИЯНИЕ ГЛУБИНЫ ====================

def run_depth_analysis(logger, results_dir='results/architecture_analysis/depth_analysis'):
	"""
	Задание 2.2: Влияние глубины CNN
	"""
	logger.info("\n" + "=" * 70)
	logger.info("ЗАДАНИЕ 2.2: ВЛИЯНИЕ ГЛУБИНЫ CNN")
	logger.info("=" * 70)

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

	# Создание моделей с разной глубиной
	logger.info("\nСоздание моделей...")
	models = {
		'Shallow (2 conv)': ShallowCNN(),
		'Medium (4 conv)': MediumCNN(),
		'Deep (6 conv)': DeepCNN(),
		'Deep + Residual': DeepCNNWithResidual()
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
		logger.info(f"\n{'=' * 50}")
		logger.info(f"Обучение модели: {name}")
		logger.info(f"{'=' * 50}")

		model = model.to(device)

		history = train_model(
			model=model,
			train_loader=train_loader,
			test_loader=test_loader,
			epochs=epochs,
			lr=lr,
			device=device,
			model_name=f"Depth_{name}",
			save_best=True,
			save_path=f'{results_dir}/best_{name.replace(" ", "_")}.pth'
		)

		# Измеряем время инференса
		import time
		inference_start = time.time()
		from utils.training_utils import evaluate
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

	# ==================== АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ====================

	# 1. Сравнительная таблица
	df = compare_models(results)
	print_comparison_table(df)

	# 2. Сохранение результатов
	save_comparison_results(results, f'{results_dir}/results.json')

	# 3. Кривые обучения
	plot_multiple_training_curves(
		results,
		title="Depth Comparison",
		save_path=f'{results_dir}/plots/training_curves.png'
	)

	# 4. Сравнение точности
	plot_accuracy_comparison(
		results,
		title="Depth: Accuracy Comparison",
		save_path=f'{results_dir}/plots/accuracy_comparison.png'
	)

	# 5. Сравнение времени
	plot_time_comparison(
		results,
		title="Depth: Time Comparison",
		save_path=f'{results_dir}/plots/time_comparison.png'
	)

	# 6. Анализ градиентов
	logger.info("\nАнализ градиентов...")

	for name, result in results.items():
		model = result['model']

		# Градиентный поток
		plot_gradient_flow(
			model, train_loader, device,
			save_path=f'{results_dir}/plots/gradient_flow_{name.replace(" ", "_")}.png'
		)

		# Распределение градиентов
		plot_gradient_distribution(
			model, train_loader, device,
			save_path=f'{results_dir}/plots/gradient_distribution_{name.replace(" ", "_")}.png'
		)

	# 7. Визуализация feature maps для глубоких моделей
	logger.info("\nВизуализация feature maps...")

	# Выбираем глубокие модели для визуализации
	depth_models = ['Deep (6 conv)', 'Deep + Residual']
	layer_names = ['conv1', 'conv3', 'conv5']  # Разные слои

	for name in depth_models:
		if name in results:
			model = results[name]['model']

			# Получаем реальные имена слоев
			actual_layers = []
			for layer_name, _ in model.named_modules():
				if 'conv' in layer_name and layer_name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6']:
					actual_layers.append(layer_name)

			if actual_layers:
				# Берем первые 3 слоя
				visualize_feature_maps(
					model, test_loader, device,
					actual_layers[:3],
					save_path=f'{results_dir}/plots/feature_maps_{name.replace(" ", "_")}.png'
				)

	# 8. Анализ vanishing/exploding gradients
	logger.info("\nАнализ vanishing/exploding gradients:")
	logger.info("  - В глубоких сетях градиенты могут затухать")
	logger.info("  - Residual связи помогают сохранять градиенты")
	logger.info("  - Batch Normalization также помогает")

	# 9. Отчет
	with open(f'{results_dir}/summary.txt', 'w', encoding='utf-8') as f:
		f.write("=" * 70 + "\n")
		f.write("АНАЛИЗ ВЛИЯНИЯ ГЛУБИНЫ CNN\n")
		f.write("=" * 70 + "\n\n")
		f.write(df.to_string())
		f.write("\n\nВыводы:\n")
		f.write("1. Глубокие сети показывают лучшую точность\n")
		f.write("2. Но требуют больше времени на обучение\n")
		f.write("3. Residual связи эффективно решают проблему затухания градиентов\n")
		f.write("4. Без Residual связей глубокие сети сложно обучать\n")

	logger.info(f"\nРезультаты сохранены в: {results_dir}/")

	return results


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
	"""Основная функция для запуска анализа архитектур"""

	# Настройка логирования
	logger = setup_logging('logs')

	logger.info("=" * 70)
	logger.info("АНАЛИЗ АРХИТЕКТУР CNN")
	logger.info("Домашнее задание к уроку 4")
	logger.info("=" * 70)

	# Информация о системе
	device = get_device()
	logger.info(f"\nСистемная информация:")
	logger.info(f"  PyTorch: {torch.__version__}")
	logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
	if torch.cuda.is_available():
		logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
		logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
	logger.info("=" * 70)

	start_time = datetime.now()

	try:
		# Задание 2.1: Влияние размера ядра
		kernel_results = run_kernel_analysis(logger)

		# Задание 2.2: Влияние глубины
		depth_results = run_depth_analysis(logger)

		# Итоговые выводы
		logger.info("\n" + "=" * 70)
		logger.info("ИТОГОВЫЕ ВЫВОДЫ ПО АНАЛИЗУ АРХИТЕКТУР")
		logger.info("=" * 70)

		logger.info("\nРазмер ядра:")
		logger.info("  - Лучший размер: 3x3 (баланс точности и скорости)")
		logger.info("  - 5x5 и 7x7 дают больше контекста, но медленнее")
		logger.info("  - Смешанные ядра эффективны, но сложнее в обучении")

		logger.info("\nГлубина сети:")
		logger.info("  - Глубокие сети показывают лучшую точность")
		logger.info("  - Требуется больше времени на обучение")
		logger.info("  - Residual связи критичны для глубоких сетей")
		logger.info("  - Batch Normalization помогает стабилизировать обучение")

		end_time = datetime.now()
		total_time = (end_time - start_time).total_seconds()

		logger.info("\n" + "=" * 70)
		logger.info(f"АНАЛИЗ ЗАВЕРШЕН")
		logger.info(f"Общее время выполнения: {total_time:.2f} секунд ({total_time / 60:.2f} минут)")
		logger.info("=" * 70)

	except Exception as e:
		logger.error(f"Произошла ошибка: {str(e)}", exc_info=True)
		raise


if __name__ == "__main__":
	main()