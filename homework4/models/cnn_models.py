"""
Сверточные модели для задач компьютерного зрения
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== ВСПОМОГАТЕЛЬНЫЕ БЛОКИ ====================

class ResidualBlock(nn.Module):
	"""
	Residual блок для CNN
	"""

	def __init__(self, in_channels, out_channels, stride=1):
		super(ResidualBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x):
		residual = self.shortcut(x)
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		out = F.relu(out)
		return out


class ResidualBlockMNIST(nn.Module):
	"""
	Residual блок специально для MNIST (без изменения размерности)
	"""

	def __init__(self, channels):
		super(ResidualBlockMNIST, self).__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(channels)

	def forward(self, x):
		residual = x
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		out = F.relu(out)
		return out


class ResidualBlockCIFAR(nn.Module):
	"""
	Residual блок для CIFAR-10 (с возможностью изменения размерности)
	"""

	def __init__(self, in_channels, out_channels, stride=1):
		super(ResidualBlockCIFAR, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
				nn.BatchNorm2d(out_channels)
			)

	def forward(self, x):
		residual = self.shortcut(x)
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += residual
		out = F.relu(out)
		return out


# ==================== CNN МОДЕЛИ ДЛЯ MNIST ====================

class SimpleCNN_MNIST(nn.Module):
	"""
	Простая CNN для MNIST (2-3 сверточных слоя)
	"""

	def __init__(self, num_classes=10):
		super(SimpleCNN_MNIST, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(0.25)

		self.fc1 = nn.Linear(128 * 3 * 3, 256)
		self.fc2 = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = x.view(x.size(0), -1)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.fc2(x)
		return x


class ResCNN_MNIST(nn.Module):
	"""
	CNN с Residual блоками для MNIST
	"""

	def __init__(self, num_classes=10):
		super(ResCNN_MNIST, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(32)

		self.res_block1 = ResidualBlockMNIST(32)
		self.res_block2 = ResidualBlockMNIST(32)

		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(64)

		self.res_block3 = ResidualBlockMNIST(64)
		self.res_block4 = ResidualBlockMNIST(64)

		self.pool = nn.MaxPool2d(2, 2)
		self.dropout = nn.Dropout(0.3)
		self.fc = nn.Linear(64 * 7 * 7, num_classes)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.pool(x)
		x = self.res_block1(x)
		x = self.res_block2(x)

		x = F.relu(self.bn2(self.conv2(x)))
		x = self.pool(x)
		x = self.res_block3(x)
		x = self.res_block4(x)

		x = x.view(x.size(0), -1)
		x = self.dropout(x)
		x = self.fc(x)
		return x


class DeepCNN_MNIST(nn.Module):
	"""
	Глубокая CNN для MNIST (6 сверточных слоев)
	"""

	def __init__(self, num_classes=10):
		super(DeepCNN_MNIST, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, 3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),

			nn.Conv2d(32, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),

			nn.Conv2d(64, 128, 3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
		)

		self.classifier = nn.Sequential(
			nn.Dropout(0.3),
			nn.Linear(128 * 3 * 3, 512),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(512, 256),
			nn.ReLU(),
			nn.Linear(256, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


# ==================== CNN МОДЕЛИ ДЛЯ CIFAR-10 ====================

class SimpleCNN_CIFAR10(nn.Module):
	"""
	Простая CNN для CIFAR-10
	"""

	def __init__(self, num_classes=10):
		super(SimpleCNN_CIFAR10, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(128 * 4 * 4, 256)
		self.fc2 = nn.Linear(256, num_classes)
		self.dropout = nn.Dropout(0.25)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x


class ResCNN_CIFAR10(nn.Module):
	"""
	CNN с Residual блоками для CIFAR-10
	"""

	def __init__(self, num_classes=10):
		super(ResCNN_CIFAR10, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(64)

		self.layer1 = self._make_layer(64, 64, 2, stride=1)
		self.layer2 = self._make_layer(64, 128, 2, stride=2)
		self.layer3 = self._make_layer(128, 256, 2, stride=2)
		self.layer4 = self._make_layer(256, 512, 2, stride=2)

		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512, num_classes)
		self.dropout = nn.Dropout(0.3)

	def _make_layer(self, in_channels, out_channels, num_blocks, stride):
		layers = []
		layers.append(ResidualBlockCIFAR(in_channels, out_channels, stride))
		for _ in range(1, num_blocks):
			layers.append(ResidualBlockCIFAR(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.dropout(x)
		x = self.fc(x)
		return x


class ResCNN_CIFAR10_Regularized(nn.Module):
	"""
	CNN с регуляризацией и Residual блоками для CIFAR-10
	"""

	def __init__(self, num_classes=10):
		super(ResCNN_CIFAR10_Regularized, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(64)

		self.layer1 = self._make_layer(64, 64, 2, stride=1)
		self.layer2 = self._make_layer(64, 128, 2, stride=2)
		self.layer3 = self._make_layer(128, 256, 2, stride=2)
		self.layer4 = self._make_layer(256, 512, 2, stride=2)

		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512, num_classes)
		self.dropout = nn.Dropout(0.5)

	def _make_layer(self, in_channels, out_channels, num_blocks, stride):
		layers = []
		layers.append(ResidualBlockCIFAR(in_channels, out_channels, stride))
		for _ in range(1, num_blocks):
			layers.append(ResidualBlockCIFAR(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.dropout(x)
		x = self.fc(x)
		return x


class WideResNet_CIFAR10(nn.Module):
	"""
	Wide ResNet для CIFAR-10
	"""

	def __init__(self, num_classes=10, width=2):
		super(WideResNet_CIFAR10, self).__init__()
		base_channels = 64
		self.conv1 = nn.Conv2d(3, base_channels * width, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(base_channels * width)

		self.layer1 = self._make_layer(base_channels * width, base_channels * width, 3, stride=1)
		self.layer2 = self._make_layer(base_channels * width, base_channels * 2 * width, 3, stride=2)
		self.layer3 = self._make_layer(base_channels * 2 * width, base_channels * 4 * width, 3, stride=2)

		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(base_channels * 4 * width, num_classes)

	def _make_layer(self, in_channels, out_channels, num_blocks, stride):
		layers = [ResidualBlockCIFAR(in_channels, out_channels, stride)]
		for _ in range(1, num_blocks):
			layers.append(ResidualBlockCIFAR(out_channels, out_channels))
		return nn.Sequential(*layers)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.avg_pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x


# ==================== ФАБРИКА МОДЕЛЕЙ ====================

def get_cnn_model(model_name, dataset='mnist', num_classes=10, **kwargs):
	"""
	Фабрика для создания CNN моделей

	Args:
		model_name: имя модели
		dataset: 'mnist' или 'cifar10'
		num_classes: количество классов
		**kwargs: дополнительные параметры

	Returns:
		nn.Module: модель
	"""
	if dataset == 'mnist':
		models = {
			'simple': SimpleCNN_MNIST,
			'resnet': ResCNN_MNIST,
			'deep': DeepCNN_MNIST
		}
	elif dataset == 'cifar10':
		models = {
			'simple': SimpleCNN_CIFAR10,
			'resnet': ResCNN_CIFAR10,
			'resnet_reg': ResCNN_CIFAR10_Regularized,
			'wide_resnet': WideResNet_CIFAR10
		}
	else:
		raise ValueError(f"Unknown dataset: {dataset}")

	model_class = models.get(model_name)
	if model_class is None:
		raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

	return model_class(num_classes=num_classes, **kwargs)