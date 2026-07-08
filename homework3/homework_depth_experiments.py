# homework_depth_experiments.py - упрощенная версия
import torch
import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

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


def create_model_by_depth(num_hidden_layers, hidden_size=256, use_dropout=False,
                          use_batchnorm=False, input_size=784, num_classes=10):
	"""Создает модель с заданным количеством скрытых слоев"""
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


def save_results(results, dataset_name):
	"""Сохраняет результаты экспериментов в JSON файл"""
	results_dir = 'results/depth_experiments'
	os.makedirs(results_dir, exist_ok=True)

	save_data = {}
	for key, value in results.items():
		save_data[key] = {
			'train_accs': value['train_accs'],
			'test_accs': value['test_accs'],
			'training_time': value['training_time'],
			'parameters': value['parameters'],
			'config_name': value['config_name'],
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


def plot_depth_comparison(results, dataset_name):
	"""Визуализирует сравнение моделей разной глубины"""
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))

	depths = sorted([int(k) for k in results.keys()])

	# График точности
	ax = axes[0, 0]
	train_accs = [results[str(d)]['train_accs'][-1] for d in depths]
	test_accs = [results[str(d)]['test_accs'][-1] for d in depths]
	ax.plot(depths, train_accs, 'o-', label='Train', linewidth=2, markersize=8)
	ax.plot(depths, test_accs, 's-', label='Test', linewidth=2, markersize=8)
	ax.set_xlabel('Number of Hidden Layers')
	ax.set_ylabel('Accuracy')
	ax.set_title('Final Accuracy vs Depth')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# График времени
	ax = axes[0, 1]
	times = [results[str(d)]['training_time'] for d in depths]
	ax.bar(depths, times)
	ax.set_xlabel('Number of Hidden Layers')
	ax.set_ylabel('Training Time (seconds)')
	ax.set_title('Training Time vs Depth')
	ax.grid(True, alpha=0.3)

	# График параметров
	ax = axes[1, 0]
	params = [results[str(d)]['parameters'] for d in depths]
	ax.bar(depths, params)
	ax.set_xlabel('Number of Hidden Layers')
	ax.set_ylabel('Number of Parameters')
	ax.set_title('Model Size vs Depth')
	ax.grid(True, alpha=0.3)

	# График переобучения
	ax = axes[1, 1]
	overfitting = [abs(results[str(d)]['train_accs'][-1] - results[str(d)]['test_accs'][-1])
	               for d in depths]
	ax.plot(depths, overfitting, 'o-', color='red', linewidth=2, markersize=8)
	ax.set_xlabel('Number of Hidden Layers')
	ax.set_ylabel('Train-Test Accuracy Gap')
	ax.set_title('Overfitting vs Depth')
	ax.grid(True, alpha=0.3)

	plt.suptitle(f'{dataset_name}: Depth Analysis', fontsize=14, fontweight='bold')
	plt.tight_layout()

	# Сохраняем
	os.makedirs('plots/depth', exist_ok=True)
	plt.savefig(f'plots/depth/{dataset_name.lower()}_depth_analysis.png',
	            dpi=300, bbox_inches='tight')
	plt.show()

	return fig


def plot_training_curves(results, dataset_name):
	"""Визуализирует кривые обучения"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

	depths = sorted([int(k) for k in results.keys()])
	colors = plt.cm.tab10(np.linspace(0, 1, len(depths)))

	# Accuracy
	for i, depth in enumerate(depths):
		history = results[str(depth)]['history']
		epochs = range(1, len(history['train_accs']) + 1)
		ax1.plot(epochs, history['train_accs'], '--',
		         color=colors[i], label=f'{depth} layers (train)', alpha=0.7)
		ax1.plot(epochs, history['test_accs'], '-',
		         color=colors[i], label=f'{depth} layers (test)', linewidth=2)

	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Accuracy')
	ax1.set_title('Training Progress')
	ax1.legend(loc='best', fontsize=8)
	ax1.grid(True, alpha=0.3)

	# Loss
	for i, depth in enumerate(depths):
		history = results[str(depth)]['history']
		epochs = range(1, len(history['train_losses']) + 1)
		ax2.plot(epochs, history['train_losses'], '--',
		         color=colors[i], label=f'{depth} layers (train)', alpha=0.7)
		ax2.plot(epochs, history['test_losses'], '-',
		         color=colors[i], label=f'{depth} layers (test)', linewidth=2)

	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Loss')
	ax2.set_title('Loss Curves')
	ax2.legend(loc='best', fontsize=8)
	ax2.grid(True, alpha=0.3)

	plt.suptitle(f'{dataset_name}: Training Curves', fontsize=14, fontweight='bold')
	plt.tight_layout()

	# Сохраняем
	os.makedirs('plots/depth', exist_ok=True)
	plt.savefig(f'plots/depth/{dataset_name.lower()}_training_curves.png',
	            dpi=300, bbox_inches='tight')
	plt.show()

	return fig


def generate_report(results, dataset_name):
	"""Генерирует отчет по экспериментам"""
	print(f"\n{'=' * 50}")
	print(f"REPORT: {dataset_name}")
	print(f"{'=' * 50}\n")

	print("Performance Summary:")
	print("-" * 70)
	print(f"{'Config':<25} {'Train Acc':<12} {'Test Acc':<12} {'Params':<15} {'Time (s)':<10}")
	print("-" * 70)

	best_depth = None
	best_acc = 0

	for depth in sorted([int(k) for k in results.keys()]):
		res = results[str(depth)]
		print(f"{res['config_name']:<25} {res['train_accs'][-1]:<12.4f} "
		      f"{res['test_accs'][-1]:<12.4f} {res['parameters']:<15,} "
		      f"{res['training_time']:<10.2f}")

		if res['test_accs'][-1] > best_acc:
			best_acc = res['test_accs'][-1]
			best_depth = depth

	print("-" * 70)
	print(f"\nBest: {best_depth} hidden layers with {best_acc:.4f} test accuracy")
	print()

	# Сохраняем отчет
	os.makedirs('results/depth_experiments', exist_ok=True)
	with open(f'results/depth_experiments/{dataset_name.lower()}_report.txt', 'w') as f:
		f.write(f"REPORT: {dataset_name}\n")
		f.write("=" * 50 + "\n\n")
		f.write("Performance Summary:\n")
		f.write("-" * 70 + "\n")
		f.write(f"{'Config':<25} {'Train Acc':<12} {'Test Acc':<12} {'Params':<15} {'Time (s)':<10}\n")
		f.write("-" * 70 + "\n")

		for depth in sorted([int(k) for k in results.keys()]):
			res = results[str(depth)]
			f.write(f"{res['config_name']:<25} {res['train_accs'][-1]:<12.4f} "
			        f"{res['test_accs'][-1]:<12.4f} {res['parameters']:<15,} "
			        f"{res['training_time']:<10.2f}\n")

		f.write("-" * 70 + "\n")
		f.write(f"\nBest: {best_depth} hidden layers with {best_acc:.4f} test accuracy\n")


def run_depth_experiments(dataset_name, train_loader, test_loader,
                          hidden_size=256, epochs=5):
	"""Запускает эксперименты с разной глубиной сети"""
	print(f"\n{'=' * 50}")
	print(f"Depth Experiments on {dataset_name}")
	print(f"{'=' * 50}\n")

	configs = [
		{'name': 'Shallow (0 hidden)', 'hidden_layers': 0},
		{'name': 'Medium (2 hidden)', 'hidden_layers': 2},
		{'name': 'Deep (4 hidden)', 'hidden_layers': 4}
	]

	results = {}

	for config in configs:
		hidden_layers = config['hidden_layers']
		name = config['name']

		print(f"Training {name}...")

		if hidden_layers == 0:
			model = create_shallow_model(input_size=784, num_classes=10).to(device)
		else:
			model = create_model_by_depth(
				num_hidden_layers=hidden_layers,
				hidden_size=hidden_size,
				use_dropout=False,
				use_batchnorm=False,
				input_size=784,
				num_classes=10
			).to(device)

		history, training_time = run_experiment(
			model, train_loader, test_loader,
			epochs=epochs, lr=0.001,
			model_name=f"Depth {hidden_layers}",
			verbose=True
		)

		results[str(hidden_layers)] = {
			'history': history,
			'training_time': training_time,
			'parameters': count_parameters(model),
			'train_accs': history['train_accs'],
			'test_accs': history['test_accs'],
			'config_name': name
		}

	return results


def main():
	"""Основная функция"""
	print("=" * 50)
	print("DEPTH EXPERIMENTS")
	print("=" * 50)

	# Загружаем датасеты
	print("\nLoading datasets...")
	mnist_train, mnist_test = get_mnist_loaders(batch_size=128)
	fashion_train, fashion_test = get_fashion_mnist_loaders(batch_size=128)
	print("Done!\n")

	# MNIST
	print("\n" + "=" * 50)
	print("MNIST EXPERIMENTS")
	print("=" * 50)

	mnist_results = run_depth_experiments(
		"MNIST", mnist_train, mnist_test,
		hidden_size=256, epochs=5
	)

	save_results(mnist_results, 'mnist')
	generate_report(mnist_results, 'MNIST')
	plot_depth_comparison(mnist_results, 'MNIST')
	plot_training_curves(mnist_results, 'MNIST')

	# Fashion-MNIST
	print("\n" + "=" * 50)
	print("FASHION-MNIST EXPERIMENTS")
	print("=" * 50)

	fashion_results = run_depth_experiments(
		"Fashion-MNIST", fashion_train, fashion_test,
		hidden_size=256, epochs=5
	)

	save_results(fashion_results, 'fashion')
	generate_report(fashion_results, 'Fashion-MNIST')
	plot_depth_comparison(fashion_results, 'Fashion-MNIST')
	plot_training_curves(fashion_results, 'Fashion-MNIST')

	print("\n" + "=" * 50)
	print("DEPTH EXPERIMENTS COMPLETED!")
	print("=" * 50)
	print("\nResults saved to:")
	print("  - results/depth_experiments/")
	print("  - plots/depth/")


if __name__ == "__main__":
	main()