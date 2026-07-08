# utils/model_utils.py
import sys
import os

# Добавляем родительскую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FullyConnectedModel


def create_model_by_depth(num_hidden_layers, hidden_size=256, use_dropout=False,
                          use_batchnorm=False, input_size=784, num_classes=10):
	"""
	Создает модель с заданным количеством скрытых слоев
	"""
	layers = []

	if num_hidden_layers == 0:
		return FullyConnectedModel(
			input_size=input_size,
			num_classes=num_classes,
			layers=[]
		)

	# Первый скрытый слой
	layers.append({"type": "linear", "size": hidden_size})
	if use_batchnorm:
		layers.append({"type": "batch_norm"})
	layers.append({"type": "relu"})
	if use_dropout:
		layers.append({"type": "dropout", "rate": 0.2})

	# Промежуточные скрытые слои
	for i in range(num_hidden_layers - 1):
		layers.append({"type": "linear", "size": hidden_size})
		if use_batchnorm:
			layers.append({"type": "batch_norm"})
		layers.append({"type": "relu"})
		if use_dropout:
			layers.append({"type": "dropout", "rate": 0.2})

	return FullyConnectedModel(
		input_size=input_size,
		num_classes=num_classes,
		layers=layers
	)


def create_model_by_width(hidden_size, num_hidden_layers=2, use_dropout=False,
                          use_batchnorm=False, input_size=784, num_classes=10):
	"""
	Создает модель с заданной шириной скрытых слоев
	"""
	layers = []

	if num_hidden_layers == 0:
		return FullyConnectedModel(
			input_size=input_size,
			num_classes=num_classes,
			layers=[]
		)

	# Первый скрытый слой
	layers.append({"type": "linear", "size": hidden_size})
	if use_batchnorm:
		layers.append({"type": "batch_norm"})
	layers.append({"type": "relu"})
	if use_dropout:
		layers.append({"type": "dropout", "rate": 0.2})

	# Промежуточные скрытые слои
	for i in range(num_hidden_layers - 1):
		layers.append({"type": "linear", "size": hidden_size})
		if use_batchnorm:
			layers.append({"type": "batch_norm"})
		layers.append({"type": "relu"})
		if use_dropout:
			layers.append({"type": "dropout", "rate": 0.2})

	return FullyConnectedModel(
		input_size=input_size,
		num_classes=num_classes,
		layers=layers
	)


def create_shallow_model(input_size=784, num_classes=10):
	"""Создает модель с 1 слоем (линейный классификатор)"""
	return FullyConnectedModel(
		input_size=input_size,
		num_classes=num_classes,
		layers=[]
	)


def create_model_with_regularization(num_hidden_layers, hidden_size=256,
                                     input_size=784, num_classes=10):
	"""
	Создает модель с регуляризацией (Dropout + BatchNorm)
	"""
	return create_model_by_depth(
		num_hidden_layers=num_hidden_layers,
		hidden_size=hidden_size,
		use_dropout=True,
		use_batchnorm=True,
		input_size=input_size,
		num_classes=num_classes
	)