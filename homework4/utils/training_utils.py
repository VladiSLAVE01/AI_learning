"""
Утилиты для обучения нейронных сетей
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ==================== НАСТРОЙКА УСТРОЙСТВА ====================

def get_device():
	"""
	Определяет лучшее доступное устройство (GPU/CPU)

	Returns:
		torch.device: устройство для вычислений
	"""
	if torch.cuda.is_available():
		device = torch.device("cuda")
		logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
		logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
	elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
		device = torch.device("mps")
		logger.info("Using Apple MPS (Metal Performance Shaders)")
	else:
		device = torch.device("cpu")
		logger.info("Using CPU")

	return device


def set_seed(seed=42):
	"""
	Устанавливает seed для воспроизводимости

	Args:
		seed: значение seed
	"""
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	logger.info(f"Seed set to {seed}")


# ==================== ПОДСЧЕТ ПАРАМЕТРОВ ====================

def count_parameters(model):
	"""
	Подсчитывает количество обучаемых параметров в модели

	Args:
		model: модель PyTorch

	Returns:
		int: количество параметров
	"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
	"""
	Выводит сводку по модели

	Args:
		model: модель PyTorch
	"""
	total_params = count_parameters(model)
	logger.info(f"Model: {model.__class__.__name__}")
	logger.info(f"Total parameters: {total_params:,}")
	logger.info(f"Layers: {len(list(model.modules()))}")

	# Подсчет параметров по слоям
	layer_params = []
	for name, param in model.named_parameters():
		if param.requires_grad:
			layer_params.append((name, param.numel()))

	if layer_params:
		logger.info("Parameters per layer:")
		for name, count in layer_params[:10]:  # Показываем первые 10 слоев
			logger.info(f"  {name}: {count:,}")
		if len(layer_params) > 10:
			logger.info(f"  ... and {len(layer_params) - 10} more layers")


# ==================== ОБУЧЕНИЕ ====================

def train_epoch(model, dataloader, criterion, optimizer, device,
                scheduler=None, accumulation_steps=1):
	"""
	Обучение одной эпохи

	Args:
		model: модель
		dataloader: DataLoader для обучения
		criterion: функция потерь
		optimizer: оптимизатор
		device: устройство
		scheduler: планировщик обучения (опционально)
		accumulation_steps: шаги накопления градиента

	Returns:
		tuple: (средняя потеря, точность)
	"""
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	progress_bar = tqdm(dataloader, desc="Training", leave=False)

	for batch_idx, (data, target) in enumerate(progress_bar):
		data, target = data.to(device), target.to(device)

		# Forward pass
		output = model(data)
		loss = criterion(output, target)
		loss = loss / accumulation_steps  # Normalize for accumulation

		# Backward pass
		loss.backward()

		# Update weights
		if (batch_idx + 1) % accumulation_steps == 0:
			optimizer.step()
			optimizer.zero_grad()
			if scheduler:
				scheduler.step()

		# Statistics
		running_loss += loss.item() * accumulation_steps
		_, predicted = output.max(1)
		total += target.size(0)
		correct += predicted.eq(target).sum().item()

		# Update progress bar
		progress_bar.set_postfix({
			'loss': running_loss / (batch_idx + 1),
			'acc': 100. * correct / total
		})

	epoch_loss = running_loss / len(dataloader)
	epoch_acc = 100. * correct / total

	return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, return_predictions=False):
	"""
	Оценка модели на данных

	Args:
		model: модель
		dataloader: DataLoader для оценки
		criterion: функция потерь
		device: устройство
		return_predictions: возвращать предсказания

	Returns:
		tuple: (потеря, точность, [предсказания, метки]) если return_predictions=True
	"""
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	all_preds = []
	all_targets = []

	with torch.no_grad():
		for data, target in tqdm(dataloader, desc="Evaluating", leave=False):
			data, target = data.to(device), target.to(device)

			output = model(data)
			loss = criterion(output, target)

			running_loss += loss.item()
			_, predicted = output.max(1)
			total += target.size(0)
			correct += predicted.eq(target).sum().item()

			if return_predictions:
				all_preds.extend(predicted.cpu().numpy())
				all_targets.extend(target.cpu().numpy())

	epoch_loss = running_loss / len(dataloader)
	epoch_acc = 100. * correct / total

	if return_predictions:
		return epoch_loss, epoch_acc, all_preds, all_targets

	return epoch_loss, epoch_acc


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001,
                device=None, criterion=None, optimizer=None, scheduler=None,
                weight_decay=0, model_name="Model", save_best=True,
                save_path=None, logger=None):
	"""
	Полный цикл обучения модели

	Args:
		model: модель
		train_loader: DataLoader для обучения
		test_loader: DataLoader для тестирования
		epochs: количество эпох
		lr: скорость обучения
		device: устройство
		criterion: функция потерь
		optimizer: оптимизатор
		scheduler: планировщик обучения
		weight_decay: L2 регуляризация
		model_name: имя модели для логирования
		save_best: сохранять лучшую модель
		save_path: путь для сохранения
		logger: логгер

	Returns:
		dict: история обучения
	"""
	if logger is None:
		logger = logging.getLogger(__name__)

	if device is None:
		device = get_device()

	if criterion is None:
		criterion = nn.CrossEntropyLoss()

	if optimizer is None:
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

	model = model.to(device)

	logger.info(f"\n{'=' * 60}")
	logger.info(f"Training {model_name}")
	logger.info(f"Parameters: {count_parameters(model):,}")
	logger.info(f"Learning rate: {lr}")
	if weight_decay > 0:
		logger.info(f"Weight decay: {weight_decay}")
	logger.info(f"Device: {device}")
	logger.info(f"{'=' * 60}")

	# История обучения
	history = {
		'train_losses': [],
		'train_accs': [],
		'test_losses': [],
		'test_accs': [],
		'best_test_acc': 0.0,
		'best_epoch': 0
	}

	start_time = time.time()
	best_model_state = None

	for epoch in range(1, epochs + 1):
		# Обучение
		train_loss, train_acc = train_epoch(
			model, train_loader, criterion, optimizer, device, scheduler
		)

		# Оценка
		test_loss, test_acc = evaluate(model, test_loader, criterion, device)

		# Сохранение истории
		history['train_losses'].append(train_loss)
		history['train_accs'].append(train_acc)
		history['test_losses'].append(test_loss)
		history['test_accs'].append(test_acc)

		# Логирование
		logger.info(f'Epoch {epoch}/{epochs}:')
		logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
		logger.info(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

		# Сохранение лучшей модели
		if test_acc > history['best_test_acc']:
			history['best_test_acc'] = test_acc
			history['best_epoch'] = epoch
			best_model_state = model.state_dict().copy()

			if save_best and save_path:
				os.makedirs(os.path.dirname(save_path), exist_ok=True)
				torch.save({
					'epoch': epoch,
					'model_state_dict': best_model_state,
					'optimizer_state_dict': optimizer.state_dict(),
					'test_acc': test_acc,
					'history': history
				}, save_path)
				logger.info(f'  ✓ Best model saved (Test Acc: {test_acc:.2f}%)')

		logger.info('-' * 50)

	# Загрузка лучшей модели
	if best_model_state is not None:
		model.load_state_dict(best_model_state)

	training_time = time.time() - start_time
	logger.info(f"Training completed in {training_time:.2f} seconds")
	logger.info(f"Best Test Accuracy: {history['best_test_acc']:.2f}% (Epoch {history['best_epoch']})")

	history['training_time'] = training_time
	history['best_model_state'] = best_model_state

	return history


# ==================== ОПТИМИЗАТОРЫ И ПЛАНИРОВЩИКИ ====================

def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=0, **kwargs):
	"""
	Создает оптимизатор

	Args:
		model: модель
		optimizer_name: имя оптимизатора
		lr: скорость обучения
		weight_decay: L2 регуляризация
		**kwargs: дополнительные параметры для оптимизатора

	Returns:
		torch.optim.Optimizer: оптимизатор
	"""
	optimizers = {
		'adam': optim.Adam,
		'sgd': optim.SGD,
		'adamw': optim.AdamW,
		'rmsprop': optim.RMSprop,
		'adagrad': optim.Adagrad,
		'adadelta': optim.Adadelta
	}

	opt_class = optimizers.get(optimizer_name.lower())
	if opt_class is None:
		raise ValueError(f"Unknown optimizer: {optimizer_name}")

	if optimizer_name.lower() == 'sgd':
		return opt_class(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, **kwargs)

	return opt_class(model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)


def get_scheduler(optimizer, scheduler_name='step', **kwargs):
	"""
	Создает планировщик обучения

	Args:
		optimizer: оптимизатор
		scheduler_name: имя планировщика
		**kwargs: параметры планировщика

	Returns:
		torch.optim.lr_scheduler: планировщик
	"""
	schedulers = {
		'step': optim.lr_scheduler.StepLR,
		'cosine': optim.lr_scheduler.CosineAnnealingLR,
		'cosine_warm': optim.lr_scheduler.CosineAnnealingWarmRestarts,
		'reduce': optim.lr_scheduler.ReduceLROnPlateau,
		'exponential': optim.lr_scheduler.ExponentialLR,
		'multi_step': optim.lr_scheduler.MultiStepLR
	}

	sched_class = schedulers.get(scheduler_name.lower())
	if sched_class is None:
		raise ValueError(f"Unknown scheduler: {scheduler_name}")

	return sched_class(optimizer, **kwargs)


# ==================== РАННЯЯ ОСТАНОВКА ====================

class EarlyStopping:
	"""
	Ранняя остановка для предотвращения переобучения
	"""

	def __init__(self, patience=10, min_delta=0.001, mode='min'):
		"""
		Args:
			patience: количество эпох без улучшения
			min_delta: минимальное изменение для улучшения
			mode: 'min' для потерь, 'max' для точности
		"""
		self.patience = patience
		self.min_delta = min_delta
		self.mode = mode
		self.counter = 0
		self.best_score = None
		self.early_stop = False

	def __call__(self, score):
		if self.best_score is None:
			self.best_score = score
			return False

		if self.mode == 'min':
			if score < self.best_score - self.min_delta:
				self.best_score = score
				self.counter = 0
			else:
				self.counter += 1
		else:  # 'max'
			if score > self.best_score + self.min_delta:
				self.best_score = score
				self.counter = 0
			else:
				self.counter += 1

		if self.counter >= self.patience:
			self.early_stop = True

		return self.early_stop


# ==================== СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛЕЙ ====================

def save_checkpoint(model, optimizer, epoch, history, path, best=False):
	"""
	Сохраняет чекпоинт модели

	Args:
		model: модель
		optimizer: оптимизатор
		epoch: текущая эпоха
		history: история обучения
		path: путь для сохранения
		best: сохранять как лучшую модель
	"""
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'history': history,
		'is_best': best
	}

	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(checkpoint, path)
	logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
	"""
	Загружает чекпоинт модели

	Args:
		model: модель
		optimizer: оптимизатор
		path: путь к чекпоинту

	Returns:
		tuple: (модель, оптимизатор, эпоха, история)
	"""
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	history = checkpoint['history']

	logger.info(f"Checkpoint loaded: {path} (Epoch {epoch})")
	return model, optimizer, epoch, history