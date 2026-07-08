# utils/visualization_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


def save_plot(fig, filename, plots_dir='plots', subdir=''):
	"""
	Сохраняет график в файл
	"""
	# Создаем директорию
	save_path = os.path.join(plots_dir, subdir)
	os.makedirs(save_path, exist_ok=True)

	# Полный путь к файлу
	full_path = os.path.join(save_path, filename)

	# Сохраняем
	fig.savefig(full_path, dpi=300, bbox_inches='tight')
	plt.close(fig)

	print(f"Plot saved to: {full_path}")
	return full_path


def plot_depth_comparison(results, dataset_name, save=True, plots_dir='plots'):
	"""
	Визуализирует сравнение моделей разной глубины
	"""
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
	bars = ax.bar(depths, times)
	ax.set_xlabel('Number of Hidden Layers')
	ax.set_ylabel('Training Time (seconds)')
	ax.set_title('Training Time vs Depth')
	ax.grid(True, alpha=0.3)

	# График параметров
	ax = axes[1, 0]
	params = [results[str(d)]['parameters'] for d in depths]
	bars = ax.bar(depths, params)
	ax.set_xlabel('Number of Hidden Layers')
	ax.set_ylabel('Number of Parameters')
	ax.set_title('Model Size vs Depth')
	ax.grid(True, alpha=0.3)

	# График переобучения
	ax = axes[1, 1]
	overfitting = [abs(results[str(d)]['train_accs'][-1] - results[str(d)]['test_accs'][-1]) for d in depths]
	ax.plot(depths, overfitting, 'o-', color='red', linewidth=2, markersize=8)
	ax.set_xlabel('Number of Hidden Layers')
	ax.set_ylabel('Train-Test Accuracy Gap')
	ax.set_title('Overfitting vs Depth')
	ax.grid(True, alpha=0.3)

	plt.suptitle(f'{dataset_name}: Depth Analysis', fontsize=14, fontweight='bold')
	plt.tight_layout()

	if save:
		save_plot(fig, f'{dataset_name.lower()}_depth_analysis.png', plots_dir, 'depth')

	return fig


def plot_training_curves(results, dataset_name, save=True, plots_dir='plots'):
	"""
	Визуализирует кривые обучения
	"""
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

	if save:
		save_plot(fig, f'{dataset_name.lower()}_training_curves.png', plots_dir, 'depth')

	return fig


def plot_regularization_comparison(results, dataset_name, save=True, plots_dir='plots'):
	"""
	Визуализирует сравнение с регуляризацией
	"""
	fig, axes = plt.subplots(1, 3, figsize=(12, 4))

	depths = sorted([int(k) for k in results.keys()])

	# Точность
	ax = axes[0]
	x = np.arange(len(depths))
	width = 0.35

	no_reg_accs = [results[str(d)]['no_reg']['history']['test_accs'][-1] for d in depths]
	with_reg_accs = [results[str(d)]['with_reg']['history']['test_accs'][-1] for d in depths]

	ax.bar(x - width / 2, no_reg_accs, width, label='No Reg', color='lightcoral')
	ax.bar(x + width / 2, with_reg_accs, width, label='With Reg', color='lightblue')
	ax.set_xlabel('Hidden Layers')
	ax.set_ylabel('Test Accuracy')
	ax.set_title('Test Accuracy Comparison')
	ax.set_xticks(x)
	ax.set_xticklabels(depths)
	ax.legend()
	ax.grid(True, alpha=0.3)

	# Переобучение
	ax = axes[1]
	no_reg_gap = [abs(results[str(d)]['no_reg']['history']['train_accs'][-1] -
	                  results[str(d)]['no_reg']['history']['test_accs'][-1]) for d in depths]
	with_reg_gap = [abs(results[str(d)]['with_reg']['history']['train_accs'][-1] -
	                    results[str(d)]['with_reg']['history']['test_accs'][-1]) for d in depths]

	ax.bar(x - width / 2, no_reg_gap, width, label='No Reg', color='lightcoral')
	ax.bar(x + width / 2, with_reg_gap, width, label='With Reg', color='lightblue')
	ax.set_xlabel('Hidden Layers')
	ax.set_ylabel('Train-Test Gap')
	ax.set_title('Overfitting Comparison')
	ax.set_xticks(x)
	ax.set_xticklabels(depths)
	ax.legend()
	ax.grid(True, alpha=0.3)

	# Кривые обучения
	ax = axes[2]
	depth = depths[1]  # средняя глубина
	history_no_reg = results[str(depth)]['no_reg']['history']
	history_with_reg = results[str(depth)]['with_reg']['history']

	epochs = range(1, len(history_no_reg['train_accs']) + 1)
	ax.plot(epochs, history_no_reg['test_accs'], '-', label='No Reg', linewidth=2, color='lightcoral')
	ax.plot(epochs, history_with_reg['test_accs'], '-', label='With Reg', linewidth=2, color='lightblue')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Test Accuracy')
	ax.set_title(f'Learning Curves (Depth {depth})')
	ax.legend()
	ax.grid(True, alpha=0.3)

	plt.suptitle(f'{dataset_name}: Regularization Analysis', fontsize=14, fontweight='bold')
	plt.tight_layout()

	if save:
		save_plot(fig, f'{dataset_name.lower()}_regularization.png', plots_dir, 'regularization')

	return fig


def plot_width_comparison(results, dataset_name, save=True, plots_dir='plots'):
	"""
	Визуализирует сравнение моделей разной ширины
	"""
	fig, axes = plt.subplots(2, 2, figsize=(12, 8))

	widths = sorted([int(k) for k in results.keys()])

	# График точности
	ax = axes[0, 0]
	train_accs = [results[str(w)]['train_accs'][-1] for w in widths]
	test_accs = [results[str(w)]['test_accs'][-1] for w in widths]
	ax.plot(widths, train_accs, 'o-', label='Train', linewidth=2, markersize=8)
	ax.plot(widths, test_accs, 's-', label='Test', linewidth=2, markersize=8)
	ax.set_xlabel('Hidden Layer Size')
	ax.set_ylabel('Accuracy')
	ax.set_title('Final Accuracy vs Width')
	ax.legend()
	ax.grid(True, alpha=0.3)

	# График времени
	ax = axes[0, 1]
	times = [results[str(w)]['training_time'] for w in widths]
	ax.bar(widths, times)
	ax.set_xlabel('Hidden Layer Size')
	ax.set_ylabel('Training Time (seconds)')
	ax.set_title('Training Time vs Width')
	ax.grid(True, alpha=0.3)

	# График параметров
	ax = axes[1, 0]
	params = [results[str(w)]['parameters'] for w in widths]
	ax.bar(widths, params)
	ax.set_xlabel('Hidden Layer Size')
	ax.set_ylabel('Number of Parameters')
	ax.set_title('Model Size vs Width')
	ax.grid(True, alpha=0.3)

	# График переобучения
	ax = axes[1, 1]
	overfitting = [abs(results[str(w)]['train_accs'][-1] - results[str(w)]['test_accs'][-1]) for w in widths]
	ax.plot(widths, overfitting, 'o-', color='red', linewidth=2, markersize=8)
	ax.set_xlabel('Hidden Layer Size')
	ax.set_ylabel('Train-Test Accuracy Gap')
	ax.set_title('Overfitting vs Width')
	ax.grid(True, alpha=0.3)

	plt.suptitle(f'{dataset_name}: Width Analysis', fontsize=14, fontweight='bold')
	plt.tight_layout()

	if save:
		save_plot(fig, f'{dataset_name.lower()}_width_analysis.png', plots_dir, 'width')

	return fig


def create_all_plots(results, dataset_name, experiment_type, plots_dir='plots'):
	"""
	Создает все графики для эксперимента
	"""
	plots = {}

	if experiment_type == 'depth':
		plots['analysis'] = plot_depth_comparison(results, dataset_name, True, plots_dir)
		plots['curves'] = plot_training_curves(results, dataset_name, True, plots_dir)
	elif experiment_type == 'width':
		plots['analysis'] = plot_width_comparison(results, dataset_name, True, plots_dir)
		plots['curves'] = plot_training_curves(results, dataset_name, True, plots_dir)
	elif experiment_type == 'regularization':
		plots['comparison'] = plot_regularization_comparison(results, dataset_name, True, plots_dir)

	return plots