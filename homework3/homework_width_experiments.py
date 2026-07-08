# homework_width_experiments.py
import torch
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Добавляем путь для импорта из текущей директории
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты из ваших файлов
from fully_connected_basics.datasets import get_mnist_loaders, get_fashion_mnist_loaders
from fully_connected_basics.trainer import train_model
from fully_connected_basics.models import FullyConnectedModel
from fully_connected_basics.utils import count_parameters, plot_training_history

# Настройка устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def create_model_with_widths(widths, input_size=784, num_classes=10,
                             use_dropout=False, use_batchnorm=False):
	"""
	Создает модель с заданными ширинами слоев

	Args:
		widths: список размеров скрытых слоев, например [256, 128, 64]
		input_size: размер входных данных
		num_classes: количество классов
		use_dropout: использовать ли Dropout
		use_batchnorm: использовать ли BatchNorm
	"""
	layers = []

	for i, width in enumerate(widths):
		layers.append({"type": "linear", "size": width})
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


def run_experiment(model, train_loader, test_loader, epochs=5, lr=0.001,
                   model_name="model", verbose=True):
	"""Запускает обучение модели"""
	model = model.to(device)

	start_time = time.time()
	history = train_model(model, train_loader, test_loader, epochs=epochs,
	                      lr=lr, device=str(device))
	training_time = time.time() - start_time

	if verbose:
		print(f"\n{model_name} Results:")
		print(f"Final Train Acc: {history['train_accs'][-1]:.4f}")
		print(f"Final Test Acc: {history['test_accs'][-1]:.4f}")
		print(f"Time: {training_time:.2f}s")
		print(f"Params: {count_parameters(model):,}")
		print("-" * 40)

	return history, training_time


def save_results(results, dataset_name, experiment_type='width'):
	"""Сохраняет результаты экспериментов в JSON файл"""
	results_dir = f'results/{experiment_type}_experiments'
	os.makedirs(results_dir, exist_ok=True)

	save_data = {}
	for key, value in results.items():
		save_data[key] = {
			'train_accs': value['train_accs'],
			'test_accs': value['test_accs'],
			'training_time': value['training_time'],
			'parameters': value['parameters'],
			'config_name': value['config_name'],
			'widths': value.get('widths', []),
			'history': {
				'train_losses': value['history']['train_losses'],
				'test_losses': value['history']['test_losses'],
				'train_accs': value['history']['train_accs'],
				'test_accs': value['history']['test_accs']
			}
		}

	filename = os.path.join(results_dir, f'{dataset_name}_results.json')
	with open(filename, 'w') as f:
		json.dump(save_data, f, indent=2)

	print(f"Results saved to: {filename}")
	return filename


# ============= ЗАДАНИЕ 2.1: Сравнение моделей разной ширины =============

def run_width_comparison_experiments(dataset_name, train_loader, test_loader,
                                     epochs=5):
	"""
	Задание 2.1: Сравнение моделей разной ширины
	"""
	print(f"\n{'=' * 60}")
	print(f"Width Comparison Experiments on {dataset_name}")
	print(f"{'=' * 60}\n")

	# Конфигурации ширины слоев
	width_configs = [
		{
			'name': 'Narrow (64-32-16)',
			'widths': [64, 32, 16],
			'description': 'Узкие слои'
		},
		{
			'name': 'Medium (256-128-64)',
			'widths': [256, 128, 64],
			'description': 'Средние слои'
		},
		{
			'name': 'Wide (1024-512-256)',
			'widths': [1024, 512, 256],
			'description': 'Широкие слои'
		},
		{
			'name': 'Very Wide (2048-1024-512)',
			'widths': [2048, 1024, 512],
			'description': 'Очень широкие слои'
		}
	]

	results = {}

	for config in width_configs:
		name = config['name']
		widths = config['widths']

		print(f"\n--- Training {name} ---")
		print(f"Layer widths: {widths}")

		# Создаем модель
		model = create_model_with_widths(
			widths=widths,
			input_size=784,
			num_classes=10,
			use_dropout=False,
			use_batchnorm=False
		).to(device)

		# Обучаем модель
		history, training_time = run_experiment(
			model, train_loader, test_loader,
			epochs=epochs, lr=0.001,
			model_name=name,
			verbose=True
		)

		# Сохраняем результаты
		results[name] = {
			'history': history,
			'training_time': training_time,
			'parameters': count_parameters(model),
			'train_accs': history['train_accs'],
			'test_accs': history['test_accs'],
			'widths': widths,
			'config_name': name
		}

	return results


def plot_width_comparison(results, dataset_name):
	"""
	Визуализирует сравнение моделей разной ширины
	"""
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))

	names = list(results.keys())
	# Сортируем по количеству параметров
	names.sort(key=lambda x: results[x]['parameters'])

	# График точности
	ax = axes[0, 0]
	train_accs = [results[name]['train_accs'][-1] for name in names]
	test_accs = [results[name]['test_accs'][-1] for name in names]
	x = np.arange(len(names))
	width = 0.35

	ax.bar(x - width / 2, train_accs, width, label='Train', color='lightblue')
	ax.bar(x + width / 2, test_accs, width, label='Test', color='lightcoral')
	ax.set_xlabel('Model Configuration')
	ax.set_ylabel('Accuracy')
	ax.set_title('Final Accuracy vs Width')
	ax.set_xticks(x)
	ax.set_xticklabels([name.split()[0] for name in names], rotation=15)
	ax.legend()
	ax.grid(True, alpha=0.3)

	# График времени
	ax = axes[0, 1]
	times = [results[name]['training_time'] for name in names]
	bars = ax.bar(names, times)
	# Добавляем значения на столбцы
	for bar, time_val in zip(bars, times):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2., height,
		        f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
	ax.set_xlabel('Model Configuration')
	ax.set_ylabel('Training Time (seconds)')
	ax.set_title('Training Time vs Width')
	ax.set_xticklabels([name.split()[0] for name in names], rotation=15)
	ax.grid(True, alpha=0.3)

	# График параметров
	ax = axes[1, 0]
	params = [results[name]['parameters'] for name in names]
	bars = ax.bar(names, params)
	# Добавляем значения на столбцы
	for bar, param in zip(bars, params):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2., height,
		        f'{param:,}', ha='center', va='bottom', fontsize=8)
	ax.set_xlabel('Model Configuration')
	ax.set_ylabel('Number of Parameters')
	ax.set_title('Model Size vs Width')
	ax.set_xticklabels([name.split()[0] for name in names], rotation=15)
	ax.grid(True, alpha=0.3)

	# График эффективности (точность на 1 млн параметров)
	ax = axes[1, 1]
	efficiency = [results[name]['test_accs'][-1] / (results[name]['parameters'] / 1e6)
	              for name in names]
	bars = ax.bar(names, efficiency)
	for bar, eff in zip(bars, efficiency):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2., height,
		        f'{eff:.2f}', ha='center', va='bottom', fontsize=8)
	ax.set_xlabel('Model Configuration')
	ax.set_ylabel('Accuracy per Million Parameters')
	ax.set_title('Parameter Efficiency')
	ax.set_xticklabels([name.split()[0] for name in names], rotation=15)
	ax.grid(True, alpha=0.3)

	plt.suptitle(f'{dataset_name}: Width Analysis', fontsize=14, fontweight='bold')
	plt.tight_layout()

	# Сохраняем
	os.makedirs('plots/width', exist_ok=True)
	plt.savefig(f'plots/width/{dataset_name.lower()}_width_analysis.png',
	            dpi=300, bbox_inches='tight')
	plt.show()

	return fig


def plot_training_curves_width(results, dataset_name):
	"""
	Визуализирует кривые обучения для моделей разной ширины
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

	names = list(results.keys())
	names.sort(key=lambda x: results[x]['parameters'])
	colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

	# Accuracy
	for i, name in enumerate(names):
		history = results[name]['history']
		epochs = range(1, len(history['train_accs']) + 1)
		ax1.plot(epochs, history['train_accs'], '--',
		         color=colors[i], label=f'{name} (train)', alpha=0.5)
		ax1.plot(epochs, history['test_accs'], '-',
		         color=colors[i], label=f'{name} (test)', linewidth=2)

	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Accuracy')
	ax1.set_title('Training Progress by Width')
	ax1.legend(loc='best', fontsize=8)
	ax1.grid(True, alpha=0.3)

	# Loss
	for i, name in enumerate(names):
		history = results[name]['history']
		epochs = range(1, len(history['train_losses']) + 1)
		ax2.plot(epochs, history['train_losses'], '--',
		         color=colors[i], label=f'{name} (train)', alpha=0.5)
		ax2.plot(epochs, history['test_losses'], '-',
		         color=colors[i], label=f'{name} (test)', linewidth=2)

	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Loss')
	ax2.set_title('Loss Curves by Width')
	ax2.legend(loc='best', fontsize=8)
	ax2.grid(True, alpha=0.3)

	plt.suptitle(f'{dataset_name}: Training Curves by Width', fontsize=14, fontweight='bold')
	plt.tight_layout()

	# Сохраняем
	os.makedirs('plots/width', exist_ok=True)
	plt.savefig(f'plots/width/{dataset_name.lower()}_training_curves_width.png',
	            dpi=300, bbox_inches='tight')
	plt.show()

	return fig


def generate_width_report(results, dataset_name):
	"""
	Генерирует отчет по экспериментам с шириной
	"""
	print(f"\n{'=' * 60}")
	print(f"WIDTH EXPERIMENTS REPORT: {dataset_name}")
	print(f"{'=' * 60}\n")

	print("Performance Summary by Width:")
	print("-" * 90)
	print(f"{'Configuration':<25} {'Widths':<20} {'Train Acc':<12} {'Test Acc':<12} "
	      f"{'Params':<15} {'Time (s)':<10}")
	print("-" * 90)

	best_acc = 0
	best_name = None
	best_efficiency = 0
	best_eff_name = None

	for name in sorted(results.keys()):
		res = results[name]
		widths = res['widths']
		widths_str = '-'.join(map(str, widths))

		print(f"{name:<25} {widths_str:<20} {res['train_accs'][-1]:<12.4f} "
		      f"{res['test_accs'][-1]:<12.4f} {res['parameters']:<15,} "
		      f"{res['training_time']:<10.2f}")

		if res['test_accs'][-1] > best_acc:
			best_acc = res['test_accs'][-1]
			best_name = name

		efficiency = res['test_accs'][-1] / (res['parameters'] / 1e6)
		if efficiency > best_efficiency:
			best_efficiency = efficiency
			best_eff_name = name

	print("-" * 90)
	print(f"\nBest accuracy: {best_name} with {best_acc:.4f}")
	print(f"Best parameter efficiency: {best_eff_name} ({best_efficiency:.2f} acc/M params)")
	print()

	# Анализ зависимости от ширины
	print("Width Impact Analysis:")
	print("-" * 60)

	# Группируем по ширине
	narrow_names = [n for n in results.keys() if 'Narrow' in n]
	medium_names = [n for n in results.keys() if 'Medium' in n]
	wide_names = [n for n in results.keys() if 'Wide' in n]
	very_wide_names = [n for n in results.keys() if 'Very Wide' in n]

	def get_avg_acc(names):
		if names:
			return np.mean([results[n]['test_accs'][-1] for n in names])
		return 0

	print(f"Average accuracy by width category:")
	print(f"  Narrow:   {get_avg_acc(narrow_names):.4f}")
	print(f"  Medium:   {get_avg_acc(medium_names):.4f}")
	print(f"  Wide:     {get_avg_acc(wide_names):.4f}")
	print(f"  Very Wide:{get_avg_acc(very_wide_names):.4f}")
	print()

	# Сохраняем отчет
	os.makedirs('results/width_experiments', exist_ok=True)
	with open(f'results/width_experiments/{dataset_name.lower()}_report.txt', 'w') as f:
		f.write(f"WIDTH EXPERIMENTS REPORT: {dataset_name}\n")
		f.write("=" * 60 + "\n\n")
		f.write("Performance Summary by Width:\n")
		f.write("-" * 90 + "\n")
		f.write(f"{'Configuration':<25} {'Widths':<20} {'Train Acc':<12} {'Test Acc':<12} "
		        f"{'Params':<15} {'Time (s)':<10}\n")
		f.write("-" * 90 + "\n")

		for name in sorted(results.keys()):
			res = results[name]
			widths = res['widths']
			widths_str = '-'.join(map(str, widths))
			f.write(f"{name:<25} {widths_str:<20} {res['train_accs'][-1]:<12.4f} "
			        f"{res['test_accs'][-1]:<12.4f} {res['parameters']:<15,} "
			        f"{res['training_time']:<10.2f}\n")

		f.write("-" * 90 + "\n")
		f.write(f"\nBest accuracy: {best_name} with {best_acc:.4f}\n")
		f.write(f"Best parameter efficiency: {best_eff_name} ({best_efficiency:.2f} acc/M params)\n")


# ============= ЗАДАНИЕ 2.2: Оптимизация архитектуры =============

def run_architecture_search(dataset_name, train_loader, test_loader, epochs=3):
	"""
	Задание 2.2: Grid search для поиска оптимальной архитектуры
	"""
	print(f"\n{'=' * 60}")
	print(f"Architecture Search (Grid Search) on {dataset_name}")
	print(f"{'=' * 60}\n")

	# Определяем возможные ширины для каждого слоя
	width_options = [64, 128, 256, 512, 1024]

	# Определяем схемы изменения ширины
	patterns = [
		('expanding', lambda x: sorted(x)),  # расширение
		('narrowing', lambda x: sorted(x, reverse=True)),  # сужение
		('constant', lambda x: [x[0]] * 3),  # постоянная
		('symmetric', lambda x: [x[0], x[1], x[0]]),  # симметричная
	]

	results = {}

	# Grid search по комбинациям
	print("Running grid search...")
	print(f"Testing {len(width_options)} options for each layer\n")

	for i, width1 in enumerate(width_options):
		for j, width2 in enumerate(width_options):
			for k, width3 in enumerate(width_options):
				widths = [width1, width2, width3]

				# Пропускаем слишком большие комбинации (для скорости)
				if sum(widths) > 3000:
					continue

				config_name = f"{width1}-{width2}-{width3}"

				print(f"Testing {config_name}...", end=' ')

				# Создаем модель
				model = create_model_with_widths(
					widths=widths,
					input_size=784,
					num_classes=10,
					use_dropout=False,
					use_batchnorm=False
				).to(device)

				# Обучаем модель (меньше эпох для grid search)
				history, training_time = run_experiment(
					model, train_loader, test_loader,
					epochs=epochs, lr=0.001,
					model_name=config_name,
					verbose=False
				)

				# Сохраняем результаты
				results[config_name] = {
					'history': history,
					'training_time': training_time,
					'parameters': count_parameters(model),
					'train_accs': history['train_accs'],
					'test_accs': history['test_accs'],
					'widths': widths,
					'test_acc_final': history['test_accs'][-1]
				}

				print(f"Test Acc: {history['test_accs'][-1]:.4f}")

	print(f"\nGrid search completed! Tested {len(results)} configurations")
	return results


def plot_architecture_heatmap(results, dataset_name):
	"""
	Визуализирует результаты grid search в виде heatmap
	"""
	# Извлекаем данные для heatmap
	data = []
	width_options = sorted(set(
		[w for config in results.values() for w in config['widths']]
	))

	# Создаем матрицу для heatmap (только для первого и второго слоя)
	# Фиксируем третий слой как средний
	third_layer_size = 128  # берем средний размер

	# Фильтруем конфигурации с третьим слоем = third_layer_size
	filtered_results = {
		k: v for k, v in results.items()
		if v['widths'][2] == third_layer_size
	}

	if not filtered_results:
		print("No configurations with third layer = 128, using all data")
		filtered_results = results

	# Создаем матрицу
	width_values = sorted(set(
		[w for config in filtered_results.values()
		 for w in config['widths'][:2]]
	))

	heatmap_data = np.zeros((len(width_values), len(width_values)))

	for config_name, config in filtered_results.items():
		w1, w2, w3 = config['widths']
		if w3 != third_layer_size:
			continue
		try:
			i = width_values.index(w1)
			j = width_values.index(w2)
			heatmap_data[i, j] = config['test_acc_final']
		except ValueError:
			continue

	# Создаем heatmap
	fig, ax = plt.subplots(figsize=(10, 8))

	# Маска для нулевых значений
	mask = heatmap_data == 0

	im = ax.imshow(heatmap_data, cmap='viridis', interpolation='nearest',
	               aspect='auto', vmin=0.8, vmax=0.95)

	# Настройка отображения
	ax.set_xticks(np.arange(len(width_values)))
	ax.set_yticks(np.arange(len(width_values)))
	ax.set_xticklabels(width_values)
	ax.set_yticklabels(width_values)

	# Добавляем значения в ячейки
	for i in range(len(width_values)):
		for j in range(len(width_values)):
			if heatmap_data[i, j] > 0:
				text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
				               ha="center", va="center", color="white", fontsize=8)

	ax.set_xlabel('Layer 1 Width')
	ax.set_ylabel('Layer 2 Width')
	ax.set_title(f'Architecture Search Heatmap\n(Layer 3 fixed at {third_layer_size})')

	plt.colorbar(im, ax=ax, label='Test Accuracy')
	plt.tight_layout()

	# Сохраняем
	os.makedirs('plots/width', exist_ok=True)
	plt.savefig(f'plots/width/{dataset_name.lower()}_architecture_heatmap.png',
	            dpi=300, bbox_inches='tight')
	plt.show()

	return fig


def plot_architecture_patterns(results, dataset_name):
	"""
	Визуализирует сравнение разных архитектурных паттернов
	"""
	# Группируем по паттернам
	patterns = {
		'expanding': [],
		'narrowing': [],
		'constant': [],
		'symmetric': []
	}

	for config_name, config in results.items():
		widths = config['widths']
		if widths[0] < widths[1] < widths[2]:
			patterns['expanding'].append(config)
		elif widths[0] > widths[1] > widths[2]:
			patterns['narrowing'].append(config)
		elif widths[0] == widths[1] == widths[2]:
			patterns['constant'].append(config)
		elif widths[0] == widths[2] and widths[0] != widths[1]:
			patterns['symmetric'].append(config)

	# Вычисляем среднюю точность для каждого паттерна
	avg_accs = {}
	for pattern, configs in patterns.items():
		if configs:
			avg_accs[pattern] = np.mean([c['test_acc_final'] for c in configs])
		else:
			avg_accs[pattern] = 0

	# Создаем график
	fig, ax = plt.subplots(figsize=(10, 6))

	patterns_names = list(avg_accs.keys())
	accuracies = [avg_accs[p] for p in patterns_names]

	bars = ax.bar(patterns_names, accuracies, color=['lightblue', 'lightcoral', 'lightgreen', 'lightyellow'])

	# Добавляем значения
	for bar, acc in zip(bars, accuracies):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2., height,
		        f'{acc:.4f}', ha='center', va='bottom', fontsize=10)

	ax.set_xlabel('Architecture Pattern')
	ax.set_ylabel('Average Test Accuracy')
	ax.set_title(f'{dataset_name}: Architecture Pattern Comparison')
	ax.grid(True, alpha=0.3, axis='y')

	# Добавляем описание паттернов
	descriptions = {
		'expanding': 'Width increases (e.g., 64→128→256)',
		'narrowing': 'Width decreases (e.g., 256→128→64)',
		'constant': 'All layers same width (e.g., 128→128→128)',
		'symmetric': 'Symmetric (e.g., 128→256→128)'
	}

	for i, pattern in enumerate(patterns_names):
		if pattern in descriptions:
			ax.text(i, 0.02, descriptions[pattern], ha='center', va='bottom',
			        fontsize=8, rotation=0, transform=ax.get_xaxis_transform())

	plt.tight_layout()

	# Сохраняем
	os.makedirs('plots/width', exist_ok=True)
	plt.savefig(f'plots/width/{dataset_name.lower()}_architecture_patterns.png',
	            dpi=300, bbox_inches='tight')
	plt.show()

	return fig


def find_optimal_architecture(results):
	"""
	Находит оптимальную архитектуру из результатов grid search
	"""
	print("\n" + "=" * 60)
	print("OPTIMAL ARCHITECTURE FOUND")
	print("=" * 60)

	# Находим лучшую по точности
	best_acc = max(results.items(), key=lambda x: x[1]['test_acc_final'])

	# Находим лучшую по эффективности (точность на параметр)
	best_efficiency = max(results.items(),
	                      key=lambda x: x[1]['test_acc_final'] / (x[1]['parameters'] / 1e6))

	# Находим лучший компромисс (точность - параметры)
	best_tradeoff = max(results.items(),
	                    key=lambda x: x[1]['test_acc_final'] - 0.001 * (x[1]['parameters'] / 1e6))

	print("\nTop configurations:")
	print("-" * 60)

	# Топ-5 по точности
	top5 = sorted(results.items(), key=lambda x: x[1]['test_acc_final'], reverse=True)[:5]

	print("Top 5 by accuracy:")
	for i, (name, config) in enumerate(top5, 1):
		print(f"  {i}. {name} -> Acc: {config['test_acc_final']:.4f}, "
		      f"Params: {config['parameters']:,}, Time: {config['training_time']:.2f}s")

	print("\n" + "-" * 60)
	print(f"\nBest accuracy: {best_acc[0]} with {best_acc[1]['test_acc_final']:.4f}")
	print(f"Best efficiency: {best_efficiency[0]} (acc/M params: "
	      f"{best_efficiency[1]['test_acc_final'] / (best_efficiency[1]['parameters'] / 1e6):.3f})")
	print(f"Best trade-off: {best_tradeoff[0]}")

	return {
		'best_accuracy': best_acc,
		'best_efficiency': best_efficiency,
		'best_tradeoff': best_tradeoff,
		'top5': top5
	}


# ============= ОСНОВНАЯ ФУНКЦИЯ =============

def main():
	"""Основная функция"""
	print("=" * 60)
	print("WIDTH EXPERIMENTS")
	print("=" * 60)

	# Загружаем датасеты
	print("\nLoading datasets...")
	mnist_train, mnist_test = get_mnist_loaders(batch_size=128)
	fashion_train, fashion_test = get_fashion_mnist_loaders(batch_size=128)
	print("Done!\n")

	# ============ ЗАДАНИЕ 2.1: Сравнение ширины ============
	print("\n" + "=" * 60)
	print("TASK 2.1: WIDTH COMPARISON EXPERIMENTS")
	print("=" * 60)

	# MNIST
	print("\n" + "-" * 40)
	print("MNIST Width Comparison")
	print("-" * 40)

	mnist_width_results = run_width_comparison_experiments(
		"MNIST", mnist_train, mnist_test, epochs=5
	)

	save_results(mnist_width_results, 'mnist', 'width')
	generate_width_report(mnist_width_results, 'MNIST')
	plot_width_comparison(mnist_width_results, 'MNIST')
	plot_training_curves_width(mnist_width_results, 'MNIST')

	# Fashion-MNIST
	print("\n" + "-" * 40)
	print("Fashion-MNIST Width Comparison")
	print("-" * 40)

	fashion_width_results = run_width_comparison_experiments(
		"Fashion-MNIST", fashion_train, fashion_test, epochs=5
	)

	save_results(fashion_width_results, 'fashion', 'width')
	generate_width_report(fashion_width_results, 'Fashion-MNIST')
	plot_width_comparison(fashion_width_results, 'Fashion-MNIST')
	plot_training_curves_width(fashion_width_results, 'Fashion-MNIST')

	# ============ ЗАДАНИЕ 2.2: Оптимизация архитектуры ============
	print("\n" + "=" * 60)
	print("TASK 2.2: ARCHITECTURE OPTIMIZATION (Grid Search)")
	print("=" * 60)
	print("\nNote: Grid search on full dataset may take time.")
	print("Using smaller subset for demonstration...")

	# Для grid search используем меньший batch size для ускорения
	# и только MNIST для демонстрации
	print("\nRunning grid search on MNIST (reduced epochs)...")

	mnist_search_results = run_architecture_search(
		"MNIST", mnist_train, mnist_test, epochs=3
	)

	# Визуализация результатов grid search
	plot_architecture_heatmap(mnist_search_results, 'MNIST')
	plot_architecture_patterns(mnist_search_results, 'MNIST')

	# Находим оптимальную архитектуру
	optimal = find_optimal_architecture(mnist_search_results)

	# Сохраняем результаты grid search
	save_results(mnist_search_results, 'mnist_search', 'width')

	# Также запускаем grid search на Fashion-MNIST (если позволяет время)
	print("\n" + "-" * 40)
	print("Running grid search on Fashion-MNIST...")

	fashion_search_results = run_architecture_search(
		"Fashion-MNIST", fashion_train, fashion_test, epochs=3
	)

	plot_architecture_heatmap(fashion_search_results, 'Fashion-MNIST')
	plot_architecture_patterns(fashion_search_results, 'Fashion-MNIST')

	save_results(fashion_search_results, 'fashion_search', 'width')

	print("\n" + "=" * 60)
	print("WIDTH EXPERIMENTS COMPLETED!")
	print("=" * 60)
	print("\nResults saved to:")
	print("  - results/width_experiments/")
	print("  - plots/width/")


if __name__ == "__main__":
	main()