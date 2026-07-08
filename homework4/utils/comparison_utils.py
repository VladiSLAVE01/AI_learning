"""
Утилиты для сравнения моделей
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import json

logger = logging.getLogger(__name__)

# ==================== СРАВНЕНИЕ МОДЕЛЕЙ ====================

def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Сравнивает результаты нескольких моделей

    Args:
        results: словарь с результатами моделей
            {model_name: {'final_acc': float, 'train_time': float, ...}}

    Returns:
        pd.DataFrame: таблица сравнения
    """
    comparison_data = []

    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'Params': result.get('params', 0),
            'Train Acc (%)': result.get('final_train_acc', 0),
            'Test Acc (%)': result.get('final_test_acc', 0),
            'Best Test (%)': result.get('best_test_acc', 0),
            'Train Time (s)': result.get('training_time', 0),
            'Overfitting Gap': result.get('final_train_acc', 0) - result.get('final_test_acc', 0)
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Сортировка по точности на тесте
    df = df.sort_values('Test Acc (%)', ascending=False)

    return df

def print_comparison_table(df: pd.DataFrame):
    """
    Выводит таблицу сравнения в красивом формате

    Args:
        df: DataFrame с результатами
    """
    logger.info("\n" + "="*90)
    logger.info("MODEL COMPARISON")
    logger.info("="*90)

    # Форматирование таблицы
    formatted_df = df.copy()

    # Форматирование чисел
    for col in ['Params']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:,}")

    for col in ['Train Acc (%)', 'Test Acc (%)', 'Best Test (%)', 'Overfitting Gap']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")

    for col in ['Train Time (s)']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")

    # Вывод таблицы
    logger.info(formatted_df.to_string(index=False))
    logger.info("="*90)

    # Нахождение лучшей модели
    if 'Test Acc (%)' in df.columns:
        best_model = df.iloc[0]['Model']
        best_acc = df.iloc[0]['Test Acc (%)']
        logger.info(f"🏆 Best Model: {best_model} (Test Acc: {best_acc:.2f}%)")

        # Нахождение модели с наименьшим переобучением
        if 'Overfitting Gap' in df.columns:
            best_generalization = df.loc[df['Overfitting Gap'].idxmin()]
            logger.info(f"📊 Best Generalization: {best_generalization['Model']} "
                       f"(Overfitting Gap: {best_generalization['Overfitting Gap']:.2f}%)")

def analyze_overfitting(results: Dict[str, Dict]) -> Dict:
    """
    Анализирует переобучение моделей

    Args:
        results: словарь с результатами моделей

    Returns:
        dict: анализ переобучения
    """
    analysis = {}

    for model_name, result in results.items():
        train_acc = result.get('final_train_acc', 0)
        test_acc = result.get('final_test_acc', 0)

        # Вычисляем переобучение
        overfitting_gap = train_acc - test_acc
        overfitting_ratio = test_acc / train_acc if train_acc > 0 else 0

        # Определяем степень переобучения
        if overfitting_gap < 2:
            level = "Low"
        elif overfitting_gap < 5:
            level = "Moderate"
        elif overfitting_gap < 10:
            level = "High"
        else:
            level = "Severe"

        analysis[model_name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'overfitting_gap': overfitting_gap,
            'overfitting_ratio': overfitting_ratio,
            'overfitting_level': level
        }

    # Сортировка по уровню переобучения
    sorted_analysis = dict(sorted(analysis.items(),
                                  key=lambda x: x[1]['overfitting_gap']))

    # Логирование
    logger.info("\n" + "="*60)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*60)

    for model_name, stats in sorted_analysis.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Train Acc: {stats['train_acc']:.2f}%")
        logger.info(f"  Test Acc: {stats['test_acc']:.2f}%")
        logger.info(f"  Gap: {stats['overfitting_gap']:.2f}%")
        logger.info(f"  Level: {stats['overfitting_level']}")
        logger.info("-" * 40)

    return analysis

def compute_speedup(fast_result: Dict, slow_result: Dict) -> float:
    """
    Вычисляет ускорение между двумя моделями

    Args:
        fast_result: результаты быстрой модели
        slow_result: результаты медленной модели

    Returns:
        float: коэффициент ускорения
    """
    fast_time = fast_result.get('training_time', 1)
    slow_time = slow_result.get('training_time', 1)

    if slow_time == 0:
        return float('inf')

    return fast_time / slow_time

def get_best_model(results: Dict[str, Dict], metric='best_test_acc') -> Tuple[str, Dict]:
    """
    Находит лучшую модель по заданной метрике

    Args:
        results: словарь с результатами
        metric: метрика для сравнения

    Returns:
        tuple: (имя модели, результат)
    """
    best_name = max(results, key=lambda x: results[x].get(metric, 0))
    return best_name, results[best_name]

def compute_statistics(results: Dict[str, Dict]) -> Dict:
    """
    Вычисляет статистику по результатам

    Args:
        results: словарь с результатами

    Returns:
        dict: статистика
    """
    accuracies = [r.get('final_test_acc', 0) for r in results.values()]
    params = [r.get('params', 0) for r in results.values()]
    times = [r.get('training_time', 0) for r in results.values()]

    stats = {
        'mean_accuracy': np.mean(accuracies) if accuracies else 0,
        'std_accuracy': np.std(accuracies) if accuracies else 0,
        'max_accuracy': max(accuracies) if accuracies else 0,
        'min_accuracy': min(accuracies) if accuracies else 0,
        'mean_params': np.mean(params) if params else 0,
        'total_params': sum(params) if params else 0,
        'mean_training_time': np.mean(times) if times else 0,
        'total_training_time': sum(times) if times else 0
    }

    return stats

def convert_to_serializable(obj):
    """
    Рекурсивно преобразует объекты в сериализуемый формат для JSON

    Args:
        obj: объект для преобразования

    Returns:
        сериализуемый объект
    """
    if isinstance(obj, torch.Tensor):
        # Преобразуем тензор в число или список
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (nn.Module, torch.optim.Optimizer)):
        # Пропускаем модели и оптимизаторы
        return str(type(obj).__name__)
    elif isinstance(obj, (float, int, str, bool)):
        return obj
    elif obj is None:
        return None
    else:
        try:
            # Пробуем преобразовать в строку
            return str(obj)
        except:
            return f"<{type(obj).__name__}>"

def save_comparison_results(results: Dict[str, Dict], path: str):
    """
    Сохраняет результаты сравнения в файл

    Args:
        results: словарь с результатами
        path: путь для сохранения
    """
    # Преобразуем результаты в сериализуемый формат
    serializable_results = {}
    for model_name, result in results.items():
        serializable_results[model_name] = {}
        for key, value in result.items():
            # Пропускаем модели и оптимизаторы
            if key in ['model', 'optimizer', 'scheduler']:
                serializable_results[model_name][key] = str(type(value).__name__)
            elif key == 'best_model_state':
                # Не сохраняем веса модели в JSON
                serializable_results[model_name][key] = "Model state (not saved in JSON)"
            else:
                serializable_results[model_name][key] = convert_to_serializable(value)

    # Сохраняем в JSON
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {path}")

def load_comparison_results(path: str) -> Dict:
    """
    Загружает результаты сравнения из файла

    Args:
        path: путь к файлу

    Returns:
        dict: загруженные результаты
    """
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    logger.info(f"Results loaded from: {path}")
    return results

def print_model_comparison_summary(results: Dict[str, Dict]):
    """
    Выводит краткую сводку сравнения моделей

    Args:
        results: словарь с результатами
    """
    # Создаем DataFrame
    df = compare_models(results)

    # Выводим таблицу
    print_comparison_table(df)

    # Вычисляем статистику
    stats = compute_statistics(results)

    logger.info("\n" + "="*60)
    logger.info("STATISTICS SUMMARY")
    logger.info("="*60)
    logger.info(f"Models compared: {len(results)}")
    logger.info(f"Mean Test Accuracy: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%")
    logger.info(f"Best Test Accuracy: {stats['max_accuracy']:.2f}%")
    logger.info(f"Worst Test Accuracy: {stats['min_accuracy']:.2f}%")
    logger.info(f"Total Parameters: {stats['total_params']:,}")
    logger.info(f"Mean Parameters: {stats['mean_params']:,.0f}")
    logger.info(f"Total Training Time: {stats['total_training_time']:.2f}s")
    logger.info(f"Mean Training Time: {stats['mean_training_time']:.2f}s")
    logger.info("="*60)