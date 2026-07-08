# utils/experiment_utils.py
import json
import os
import time
from datetime import datetime
import torch


def get_timestamp():
	"""Возвращает текущую метку времени для имен файлов"""
	return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(results, dataset_name, experiment_type, results_dir='results'):
	"""
	Сохраняет результаты экспериментов в JSON файл

	Args:
		results: словарь с результатами
		dataset_name: название датасета ('mnist' или 'fashion')
		experiment_type: тип эксперимента ('depth', 'width', 'regularization')
		results_dir: директория для сохранения
	"""
	# Создаем директорию
	exp_dir = os.path.join(results_dir, f'{experiment_type}_experiments')
	os.makedirs(exp_dir, exist_ok=True)

	# Подготавливаем данные для сохранения
	save_data = {}
	for key, value in results.items():
		if isinstance(value, dict):
			save_data[str(key)] = {}
			for sub_key, sub_value in value.items():
				if sub_key == 'history':
					# Сохраняем историю обучения
					save_data[str(key)][sub_key] = {
						'train_losses': sub_value['train_losses'],
						'test_losses': sub_value['test_losses'],
						'train_accs': sub_value['train_accs'],
						'test_accs': sub_value['test_accs']
					}
				else:
					save_data[str(key)][sub_key] = sub_value
		else:
			save_data[str(key)] = value

	# Сохраняем в JSON
	filename = os.path.join(exp_dir, f'{dataset_name}_results.json')
	with open(filename, 'w') as f:
		json.dump(save_data, f, indent=2)

	print(f"Results saved to: {filename}")
	return filename


def load_results(dataset_name, experiment_type, results_dir='results'):
	"""
	Загружает результаты из JSON файла
	"""
	exp_dir = os.path.join(results_dir, f'{experiment_type}_experiments')
	filename = os.path.join(exp_dir, f'{dataset_name}_results.json')

	if not os.path.exists(filename):
		raise FileNotFoundError(f"Results file not found: {filename}")

	with open(filename, 'r') as f:
		data = json.load(f)

	return data


def create_experiment_summary(results):
	"""
	Создает сводку результатов эксперимента
	"""
	summary = {}

	for key, value in results.items():
		if 'test_accs' in value:
			summary[str(key)] = {
				'train_acc': value['train_accs'][-1],
				'test_acc': value['test_accs'][-1],
				'train_loss': value['history']['train_losses'][-1],
				'test_loss': value['history']['test_losses'][-1],
				'params': value.get('parameters', 0),
				'time': value.get('training_time', 0)
			}

	return summary


def print_summary(summary, title="Experiment Summary"):
	"""
	Печатает сводку результатов в виде таблицы
	"""
	print(f"\n{'=' * 70}")
	print(f"{title}")
	print(f"{'=' * 70}")
	print(f"{'Config':<15} {'Train Acc':<12} {'Test Acc':<12} {'Params':<15} {'Time (s)':<10}")
	print(f"{'-' * 70}")

	for config, metrics in sorted(summary.items()):
		print(f"{config:<15} {metrics['train_acc']:<12.4f} {metrics['test_acc']:<12.4f} "
		      f"{metrics['params']:<15,} {metrics['time']:<10.2f}")

	print(f"{'-' * 70}")

	# Находим лучшую модель
	best_config = max(summary.items(), key=lambda x: x[1]['test_acc'])
	print(f"\nBest model: {best_config[0]} with test accuracy: {best_config[1]['test_acc']:.4f}")


def save_experiment_summary(results, dataset_name, experiment_type, results_dir='results'):
	"""
	Сохраняет сводку результатов в текстовый файл
	"""
	summary = create_experiment_summary(results)

	exp_dir = os.path.join(results_dir, f'{experiment_type}_experiments')
	os.makedirs(exp_dir, exist_ok=True)

	filename = os.path.join(exp_dir, f'{dataset_name}_summary.txt')

	with open(filename, 'w') as f:
		f.write(f"Experiment Summary: {dataset_name}\n")
		f.write(f"{'=' * 60}\n\n")
		f.write(f"{'Config':<15} {'Train Acc':<12} {'Test Acc':<12} {'Params':<15} {'Time (s)':<10}\n")
		f.write(f"{'-' * 60}\n")

		for config, metrics in sorted(summary.items()):
			f.write(f"{config:<15} {metrics['train_acc']:<12.4f} {metrics['test_acc']:<12.4f} "
			        f"{metrics['params']:<15,} {metrics['time']:<10.2f}\n")

		# Находим лучшую модель
		best_config = max(summary.items(), key=lambda x: x[1]['test_acc'])
		f.write(f"\nBest model: {best_config[0]} with test accuracy: {best_config[1]['test_acc']:.4f}\n")

		# Вычисляем переобучение
		f.write(f"\nOverfitting Analysis:\n")
		f.write(f"{'-' * 60}\n")
		for config, metrics in sorted(summary.items()):
			gap = metrics['train_acc'] - metrics['test_acc']
			f.write(f"{config:<15} Gap: {gap:.4f}\n")

	print(f"Summary saved to: {filename}")