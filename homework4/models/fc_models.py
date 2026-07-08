"""
Полносвязные модели для задач компьютерного зрения
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedMNIST(nn.Module):
	"""
	Полносвязная сеть для MNIST
	Архитектура: 3-4 слоя с dropout
	"""

	def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10, dropout_rate=0.2):
		super(FullyConnectedMNIST, self).__init__()
		layers = []
		prev_size = input_size

		for hidden_size in hidden_sizes:
			layers.append(nn.Linear(prev_size, hidden_size))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout_rate))
			prev_size = hidden_size

		layers.append(nn.Linear(prev_size, num_classes))
		self.network = nn.Sequential(*layers)

	def forward(self, x):
		x = x.view(x.size(0), -1)  # Flatten
		return self.network(x)


class FullyConnectedCIFAR10(nn.Module):
	"""
	Полносвязная сеть для CIFAR-10
	Архитектура: 4-5 слоев с dropout
	"""

	def __init__(self, input_size=3072, hidden_sizes=[1024, 512, 256, 128], num_classes=10, dropout_rate=0.3):
		super(FullyConnectedCIFAR10, self).__init__()
		layers = []
		prev_size = input_size

		for hidden_size in hidden_sizes:
			layers.append(nn.Linear(prev_size, hidden_size))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout_rate))
			prev_size = hidden_size

		layers.append(nn.Linear(prev_size, num_classes))
		self.network = nn.Sequential(*layers)

	def forward(self, x):
		x = x.view(x.size(0), -1)  # Flatten
		return self.network(x)


class FullyConnectedWithBatchNorm(nn.Module):
	"""
	Полносвязная сеть с Batch Normalization
	"""

	def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], num_classes=10):
		super(FullyConnectedWithBatchNorm, self).__init__()
		layers = []
		prev_size = input_size

		for hidden_size in hidden_sizes:
			layers.append(nn.Linear(prev_size, hidden_size))
			layers.append(nn.BatchNorm1d(hidden_size))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(0.2))
			prev_size = hidden_size

		layers.append(nn.Linear(prev_size, num_classes))
		self.network = nn.Sequential(*layers)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.network(x)


class DeepFullyConnected(nn.Module):
	"""
	Глубокая полносвязная сеть (6 слоев)
	"""

	def __init__(self, input_size=784, num_classes=10):
		super(DeepFullyConnected, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_size, 1024),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(64, num_classes)
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.network(x)


class WideFullyConnected(nn.Module):
	"""
	Широкая полносвязная сеть (много нейронов в слоях)
	"""

	def __init__(self, input_size=784, num_classes=10):
		super(WideFullyConnected, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(input_size, 2048),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(2048, 1024),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(512, num_classes)
		)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		return self.network(x)


def get_fc_model(model_name, **kwargs):
	"""
	Фабрика для создания полносвязных моделей

	Args:
		model_name: имя модели ('mnist', 'cifar10', 'deep', 'wide', 'batchnorm')
		**kwargs: дополнительные параметры

	Returns:
		nn.Module: модель
	"""
	models = {
		'mnist': FullyConnectedMNIST,
		'cifar10': FullyConnectedCIFAR10,
		'deep': DeepFullyConnected,
		'wide': WideFullyConnected,
		'batchnorm': FullyConnectedWithBatchNorm
	}

	model_class = models.get(model_name)
	if model_class is None:
		raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

	return model_class(**kwargs)