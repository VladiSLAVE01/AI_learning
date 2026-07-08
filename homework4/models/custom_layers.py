"""
Кастомные слои для экспериментов с нейросетями
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== КАСТОМНЫЕ СЛОИ ====================

class SpatialDropout(nn.Module):
	"""
	Пространственный Dropout - дропаутит целые карты признаков
	Полезен для CNN, так как сохраняет пространственную структуру
	"""

	def __init__(self, p=0.5):
		super(SpatialDropout, self).__init__()
		self.p = p

	def forward(self, x):
		if not self.training or self.p == 0:
			return x

		# x: [batch_size, channels, height, width]
		batch_size, channels, height, width = x.size()

		# Создаем маску для каналов
		mask = torch.ones(batch_size, channels, 1, 1, device=x.device)
		mask = F.dropout(mask, p=self.p, training=True)

		# Применяем маску ко всем пространственным позициям
		return x * mask


class SelfAttention(nn.Module):
	"""
	Слой самовнимания (Self-Attention)
	Позволяет модели фокусироваться на важных областях изображения
	"""

	def __init__(self, in_channels, reduction=8):
		super(SelfAttention, self).__init__()
		self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
		self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
		self.value = nn.Conv2d(in_channels, in_channels, 1)
		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		batch_size, channels, height, width = x.size()

		# Query, Key, Value
		query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
		key = self.key(x).view(batch_size, -1, height * width)
		value = self.value(x).view(batch_size, -1, height * width)

		# Attention map
		attention = torch.bmm(query, key)
		attention = F.softmax(attention, dim=-1)

		# Apply attention to value
		out = torch.bmm(value, attention.permute(0, 2, 1))
		out = out.view(batch_size, channels, height, width)

		return self.gamma * out + x


class InceptionBlock(nn.Module):
	"""
	Inception блок - параллельные свертки разных размеров
	"""

	def __init__(self, in_channels, out_channels):
		super(InceptionBlock, self).__init__()
		# Ветвь 1: 1x1 свертка
		self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)

		# Ветвь 2: 1x1 -> 3x3
		self.branch2 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels // 4, 1),
			nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1)
		)

		# Ветвь 3: 1x1 -> 5x5
		self.branch3 = nn.Sequential(
			nn.Conv2d(in_channels, out_channels // 4, 1),
			nn.Conv2d(out_channels // 4, out_channels // 4, 5, padding=2)
		)

		# Ветвь 4: MaxPool -> 1x1
		self.branch4 = nn.Sequential(
			nn.MaxPool2d(3, stride=1, padding=1),
			nn.Conv2d(in_channels, out_channels // 4, 1)
		)

	def forward(self, x):
		branch1 = self.branch1(x)
		branch2 = self.branch2(x)
		branch3 = self.branch3(x)
		branch4 = self.branch4(x)

		# Concatenate по каналам
		return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class SqueezeExcitation(nn.Module):
	"""
	SE-блок (Squeeze-and-Excitation)
	Адаптивная калибровка канальных весов
	"""

	def __init__(self, in_channels, reduction=16):
		super(SqueezeExcitation, self).__init__()
		self.squeeze = nn.AdaptiveAvgPool2d(1)
		self.excitation = nn.Sequential(
			nn.Linear(in_channels, in_channels // reduction),
			nn.ReLU(),
			nn.Linear(in_channels // reduction, in_channels),
			nn.Sigmoid()
		)

	def forward(self, x):
		batch_size, channels, height, width = x.size()

		# Squeeze
		squeeze = self.squeeze(x).view(batch_size, channels)

		# Excitation
		excitation = self.excitation(squeeze).view(batch_size, channels, 1, 1)

		# Scale
		return x * excitation


class DenseBlock(nn.Module):
	"""
	Dense блок - каждый слой связан со всеми предыдущими
	"""

	def __init__(self, in_channels, growth_rate, num_layers):
		super(DenseBlock, self).__init__()
		self.layers = nn.ModuleList()

		for i in range(num_layers):
			self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

	def _make_layer(self, in_channels, growth_rate):
		return nn.Sequential(
			nn.BatchNorm2d(in_channels),
			nn.ReLU(),
			nn.Conv2d(in_channels, growth_rate, 3, padding=1)
		)

	def forward(self, x):
		features = [x]
		for layer in self.layers:
			new_features = layer(torch.cat(features, dim=1))
			features.append(new_features)

		return torch.cat(features, dim=1)


class PixelShuffleBlock(nn.Module):
	"""
	Pixel Shuffle (upsampling) - увеличение разрешения
	"""

	def __init__(self, in_channels, out_channels, upscale_factor=2):
		super(PixelShuffleBlock, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, padding=1)
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

	def forward(self, x):
		x = self.conv(x)
		x = self.pixel_shuffle(x)
		return x


class AdaptiveInstanceNorm(nn.Module):
	"""
	Адаптивная инстанс-нормализация (AdaIN)
	"""

	def __init__(self, in_channels):
		super(AdaptiveInstanceNorm, self).__init__()
		self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False)

	def forward(self, x, style):
		# x: content, style: style reference
		content_norm = self.instance_norm(x)

		# Получаем стилевые параметры из style
		style_mean = style.mean(dim=[2, 3], keepdim=True)
		style_std = style.std(dim=[2, 3], keepdim=True)

		# Применяем стиль к контенту
		return style_std * content_norm + style_mean


class GradientReversalLayer(nn.Module):
	"""
	Слой инверсии градиента (для Domain Adaptation)
	"""

	def __init__(self, alpha=1.0):
		super(GradientReversalLayer, self).__init__()
		self.alpha = alpha

	def forward(self, x):
		# На прямом проходе ничего не делаем
		return x

	def backward(self, grad_output):
		# На обратном проходе инвертируем градиент
		return -self.alpha * grad_output


# ==================== КАСТОМНЫЕ БЛОКИ ДЛЯ КЛАССИФИКАЦИИ ====================

class ClassifierWithAttention(nn.Module):
	"""
	Классификатор с механизмом внимания
	"""

	def __init__(self, in_features, num_classes, num_heads=4):
		super(ClassifierWithAttention, self).__init__()
		self.attention = nn.MultiheadAttention(in_features, num_heads, batch_first=True)
		self.classifier = nn.Linear(in_features, num_classes)

	def forward(self, x):
		# x: [batch_size, seq_len, features]
		attn_out, _ = self.attention(x, x, x)
		# Усредняем по seq_len
		pooled = attn_out.mean(dim=1)
		return self.classifier(pooled)


class MultiScaleCNN(nn.Module):
	"""
	Многомасштабная CNN
	"""

	def __init__(self, in_channels, out_channels):
		super(MultiScaleCNN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels // 3, 1)
		self.conv3 = nn.Conv2d(in_channels, out_channels // 3, 3, padding=1)
		self.conv5 = nn.Conv2d(in_channels, out_channels // 3, 5, padding=2)

	def forward(self, x):
		out1 = self.conv1(x)
		out3 = self.conv3(x)
		out5 = self.conv5(x)
		return torch.cat([out1, out3, out5], dim=1)


# ==================== ФАБРИКА КАСТОМНЫХ СЛОЕВ ====================

def get_custom_layer(layer_name, **kwargs):
	"""
	Фабрика для создания кастомных слоев
	"""
	layers = {
		'spatial_dropout': SpatialDropout,
		'self_attention': SelfAttention,
		'inception': InceptionBlock,
		'squeeze_excitation': SqueezeExcitation,
		'dense': DenseBlock,
		'pixel_shuffle': PixelShuffleBlock,
		'adaptive_instance_norm': AdaptiveInstanceNorm,
		'multi_scale': MultiScaleCNN
	}

	layer_class = layers.get(layer_name)
	if layer_class is None:
		raise ValueError(f"Unknown layer: {layer_name}. Available: {list(layers.keys())}")

	return layer_class(**kwargs)