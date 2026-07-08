"""
Домашнее задание к уроку 4: Сверточные сети
Сравнение эффективности сверточных и полносвязных сетей на задачах компьютерного зрения

Задание 1.1: Сравнение на MNIST
- Полносвязная сеть (3-4 слоя)
- Простая CNN (2-3 conv слоя)
- CNN с Residual Block

Задание 1.2: Сравнение на CIFAR-10
- Полносвязная сеть (глубокая)
- CNN с Residual блоками
- CNN с регуляризацией и Residual блоками
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты из наших модулей
from models import (
    FullyConnectedMNIST,
    FullyConnectedCIFAR10,
    SimpleCNN_MNIST,
    ResCNN_MNIST,
    ResCNN_CIFAR10,
    ResCNN_CIFAR10_Regularized,
    get_cnn_model
)

from utils import (
    get_device,
    set_seed,
    count_parameters,
    train_model,
    evaluate,
    compare_models,
    print_comparison_table,
    analyze_overfitting,
    get_best_model,
    save_comparison_results,
    print_model_comparison_summary,
    plot_training_curves,
    plot_multiple_training_curves,
    plot_confusion_matrix,
    plot_accuracy_comparison,
    plot_time_comparison,
    plot_parameter_comparison,
    plot_full_comparison
)

from datasets import get_mnist_loaders, get_cifar_loaders

# ==================== НАСТРОЙКА ====================

def setup_logging(log_dir='logs'):
    """Настройка логирования"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'comparison_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==================== ЗАДАНИЕ 1.1: MNIST ====================

def run_mnist_comparison(logger, results_dir='results/mnist_comparison'):
    """
    Задание 1.1: Сравнение на MNIST

    Сравнивает:
    - Полносвязная сеть (3-4 слоя)
    - Простая CNN (2-3 conv слоя)
    - CNN с Residual Block
    """
    logger.info("\n" + "="*70)
    logger.info("ЗАДАНИЕ 1.1: СРАВНЕНИЕ НА MNIST")
    logger.info("="*70)

    # Создаем директорию для результатов
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    # Настройка
    device = get_device()
    set_seed(42)

    # Загрузка данных
    logger.info("\nЗагрузка данных MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Создание моделей
    logger.info("\nСоздание моделей...")
    models = {
        'FC': FullyConnectedMNIST(
            hidden_sizes=[512, 256, 128],
            dropout_rate=0.2
        ),
        'Simple CNN': SimpleCNN_MNIST(),
        'ResNet': ResCNN_MNIST()
    }

    # Вывод информации о моделях
    logger.info("\nИнформация о моделях:")
    for name, model in models.items():
        params = count_parameters(model)
        logger.info(f"  {name}: {params:,} параметров")

    # Обучение моделей
    results = {}
    epochs = 10
    lr = 0.001

    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Обучение модели: {name}")
        logger.info(f"{'='*50}")

        model = model.to(device)

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            model_name=f"MNIST_{name}",
            save_best=True,
            save_path=f'{results_dir}/best_{name.replace(" ", "_")}.pth'
        )

        # Сохраняем результаты
        results[name] = {
            'model': model,
            'history': history,
            'params': count_parameters(model),
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'best_test_acc': history.get('best_test_acc', 0),
            'best_epoch': history.get('best_epoch', 0),
            'training_time': history.get('training_time', 0),
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs']
        }

    # ==================== АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ====================

    logger.info("\n" + "="*70)
    logger.info("АНАЛИЗ РЕЗУЛЬТАТОВ MNIST")
    logger.info("="*70)

    # 1. Сравнительная таблица
    df = compare_models(results)
    print_comparison_table(df)

    # 2. Анализ переобучения
    overfitting_analysis = analyze_overfitting(results)

    # 3. Сохранение результатов в JSON
    save_comparison_results(results, f'{results_dir}/results.json')

    # 4. Кривые обучения
    logger.info("\nСоздание кривых обучения...")
    plot_multiple_training_curves(
        results,
        title="MNIST: Сравнение моделей",
        save_path=f'{results_dir}/plots/training_curves.png',
        show=False
    )

    # 5. Сравнение точности
    logger.info("Создание графика сравнения точности...")
    plot_accuracy_comparison(
        results,
        title="MNIST: Сравнение точности",
        save_path=f'{results_dir}/plots/accuracy_comparison.png',
        show=False
    )

    # 6. Сравнение времени
    logger.info("Создание графика сравнения времени...")
    plot_time_comparison(
        results,
        title="MNIST: Сравнение времени",
        save_path=f'{results_dir}/plots/time_comparison.png',
        show=False
    )

    # 7. Сравнение параметров
    logger.info("Создание графика сравнения параметров...")
    plot_parameter_comparison(
        results,
        title="MNIST: Сравнение параметров",
        save_path=f'{results_dir}/plots/parameter_comparison.png',
        show=False
    )

    # 8. Лучшая модель - Confusion Matrix
    best_name, best_result = get_best_model(results)
    best_model = best_result['model']

    logger.info(f"\nЛучшая модель на MNIST: {best_name}")
    logger.info(f"Test Accuracy: {best_result['best_test_acc']:.2f}%")

    # Оценка лучшей модели с возвратом предсказаний
    from utils.training_utils import evaluate
    _, _, all_preds, all_targets = evaluate(
        best_model, test_loader, nn.CrossEntropyLoss(), device,
        return_predictions=True
    )

    # Confusion Matrix
    classes = [str(i) for i in range(10)]
    plot_confusion_matrix(
        all_targets, all_preds, classes,
        title=f"MNIST: Confusion Matrix ({best_name})",
        save_path=f'{results_dir}/plots/confusion_matrix.png',
        show=False
    )

    # 9. Отчет о результатах
    logger.info("\n" + "="*70)
    logger.info("ИТОГОВЫЙ ОТЧЕТ MNIST")
    logger.info("="*70)

    for name, result in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Параметры: {result['params']:,}")
        logger.info(f"  Train Accuracy: {result['final_train_acc']:.2f}%")
        logger.info(f"  Test Accuracy: {result['final_test_acc']:.2f}%")
        logger.info(f"  Best Test Accuracy: {result['best_test_acc']:.2f}% (Epoch {result['best_epoch']})")
        logger.info(f"  Training Time: {result['training_time']:.2f}s")

    # Вывод лучшей модели
    logger.info(f"\n🏆 Лучшая модель на MNIST: {best_name}")
    logger.info(f"   Test Accuracy: {best_result['best_test_acc']:.2f}%")

    # Сохранение сводки
    with open(f'{results_dir}/summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MNIST: СРАВНЕНИЕ МОДЕЛЕЙ\n")
        f.write("="*70 + "\n\n")
        f.write(df.to_string())
        f.write("\n\n")
        f.write("="*70 + "\n")
        f.write(f"Лучшая модель: {best_name}\n")
        f.write(f"Test Accuracy: {best_result['best_test_acc']:.2f}%\n")
        f.write("="*70 + "\n")

    logger.info(f"\nРезультаты сохранены в: {results_dir}/")

    return results

# ==================== ЗАДАНИЕ 1.2: CIFAR-10 ====================

def run_cifar10_comparison(logger, results_dir='results/cifar_comparison'):
    """
    Задание 1.2: Сравнение на CIFAR-10

    Сравнивает:
    - Полносвязная сеть (глубокая)
    - CNN с Residual блоками
    - CNN с регуляризацией и Residual блоками
    """
    logger.info("\n" + "="*70)
    logger.info("ЗАДАНИЕ 1.2: СРАВНЕНИЕ НА CIFAR-10")
    logger.info("="*70)

    # Создаем директорию для результатов
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)

    # Настройка
    device = get_device()
    set_seed(42)

    # Загрузка данных
    logger.info("\nЗагрузка данных CIFAR-10...")
    train_loader, test_loader = get_cifar_loaders(batch_size=128)
    logger.info(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Создание моделей
    logger.info("\nСоздание моделей...")
    models = {
        'FC': FullyConnectedCIFAR10(
            hidden_sizes=[1024, 512, 256, 128],
            dropout_rate=0.3
        ),
        'ResNet': ResCNN_CIFAR10(),
        'ResNet_Reg': ResCNN_CIFAR10_Regularized()
    }

    # Вывод информации о моделях
    logger.info("\nИнформация о моделях:")
    for name, model in models.items():
        params = count_parameters(model)
        logger.info(f"  {name}: {params:,} параметров")

    # Обучение моделей
    results = {}
    epochs = 15
    lr = 0.001

    for name, model in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Обучение модели: {name}")
        logger.info(f"{'='*50}")

        model = model.to(device)

        # Для регуляризованной модели добавляем weight_decay
        weight_decay = 1e-4 if name == 'ResNet_Reg' else 0

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            weight_decay=weight_decay,
            model_name=f"CIFAR10_{name}",
            save_best=True,
            save_path=f'{results_dir}/best_{name}.pth'
        )

        # Сохраняем результаты
        results[name] = {
            'model': model,
            'history': history,
            'params': count_parameters(model),
            'final_train_acc': history['train_accs'][-1],
            'final_test_acc': history['test_accs'][-1],
            'best_test_acc': history.get('best_test_acc', 0),
            'best_epoch': history.get('best_epoch', 0),
            'training_time': history.get('training_time', 0),
            'train_losses': history['train_losses'],
            'train_accs': history['train_accs'],
            'test_losses': history['test_losses'],
            'test_accs': history['test_accs']
        }

    # ==================== АНАЛИЗ И ВИЗУАЛИЗАЦИЯ ====================

    logger.info("\n" + "="*70)
    logger.info("АНАЛИЗ РЕЗУЛЬТАТОВ CIFAR-10")
    logger.info("="*70)

    # 1. Сравнительная таблица
    df = compare_models(results)
    print_comparison_table(df)

    # 2. Анализ переобучения
    overfitting_analysis = analyze_overfitting(results)

    # 3. Сохранение результатов в JSON
    save_comparison_results(results, f'{results_dir}/results.json')

    # 4. Кривые обучения
    logger.info("\nСоздание кривых обучения...")
    plot_multiple_training_curves(
        results,
        title="CIFAR-10: Сравнение моделей",
        save_path=f'{results_dir}/plots/training_curves.png',
        show=False
    )

    # 5. Сравнение точности
    logger.info("Создание графика сравнения точности...")
    plot_accuracy_comparison(
        results,
        title="CIFAR-10: Сравнение точности",
        save_path=f'{results_dir}/plots/accuracy_comparison.png',
        show=False
    )

    # 6. Сравнение времени
    logger.info("Создание графика сравнения времени...")
    plot_time_comparison(
        results,
        title="CIFAR-10: Сравнение времени",
        save_path=f'{results_dir}/plots/time_comparison.png',
        show=False
    )

    # 7. Сравнение параметров
    logger.info("Создание графика сравнения параметров...")
    plot_parameter_comparison(
        results,
        title="CIFAR-10: Сравнение параметров",
        save_path=f'{results_dir}/plots/parameter_comparison.png',
        show=False
    )

    # 8. Лучшая модель - Confusion Matrix
    best_name, best_result = get_best_model(results)
    best_model = best_result['model']

    logger.info(f"\nЛучшая модель на CIFAR-10: {best_name}")
    logger.info(f"Test Accuracy: {best_result['best_test_acc']:.2f}%")

    # Оценка лучшей модели с возвратом предсказаний
    from utils.training_utils import evaluate
    _, _, all_preds, all_targets = evaluate(
        best_model, test_loader, nn.CrossEntropyLoss(), device,
        return_predictions=True
    )

    # Confusion Matrix
    plot_confusion_matrix(
        all_targets, all_preds, classes,
        title=f"CIFAR-10: Confusion Matrix ({best_name})",
        save_path=f'{results_dir}/plots/confusion_matrix.png',
        show=False
    )

    # 9. Анализ градиентов для лучшей модели
    logger.info("\nАнализ градиентов лучшей модели...")
    from utils.visualization_utils import plot_gradient_flow, plot_gradient_distribution

    plot_gradient_flow(
        best_model, train_loader, device,
        save_path=f'{results_dir}/plots/gradient_flow.png',
        show=False
    )

    plot_gradient_distribution(
        best_model, train_loader, device,
        save_path=f'{results_dir}/plots/gradient_distribution.png',
        show=False
    )

    # 10. Отчет о результатах
    logger.info("\n" + "="*70)
    logger.info("ИТОГОВЫЙ ОТЧЕТ CIFAR-10")
    logger.info("="*70)

    for name, result in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Параметры: {result['params']:,}")
        logger.info(f"  Train Accuracy: {result['final_train_acc']:.2f}%")
        logger.info(f"  Test Accuracy: {result['final_test_acc']:.2f}%")
        logger.info(f"  Best Test Accuracy: {result['best_test_acc']:.2f}% (Epoch {result['best_epoch']})")
        logger.info(f"  Training Time: {result['training_time']:.2f}s")

    # Вывод лучшей модели
    logger.info(f"\n🏆 Лучшая модель на CIFAR-10: {best_name}")
    logger.info(f"   Test Accuracy: {best_result['best_test_acc']:.2f}%")

    # Сохранение сводки
    with open(f'{results_dir}/summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CIFAR-10: СРАВНЕНИЕ МОДЕЛЕЙ\n")
        f.write("="*70 + "\n\n")
        f.write(df.to_string())
        f.write("\n\n")
        f.write("="*70 + "\n")
        f.write(f"Лучшая модель: {best_name}\n")
        f.write(f"Test Accuracy: {best_result['best_test_acc']:.2f}%\n")
        f.write("="*70 + "\n")

    logger.info(f"\nРезультаты сохранены в: {results_dir}/")

    return results

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    """Основная функция для запуска всех экспериментов"""

    # Настройка логирования
    logger = setup_logging('logs')

    logger.info("="*70)
    logger.info("СРАВНЕНИЕ СВЕРТОЧНЫХ И ПОЛНОСВЯЗНЫХ СЕТЕЙ")
    logger.info("Домашнее задание к уроку 4")
    logger.info("="*70)

    # Информация о системе
    device = get_device()
    logger.info(f"\nСистемная информация:")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    logger.info("="*70)

    start_time = datetime.now()

    try:
        # Задание 1.1: MNIST
        mnist_results = run_mnist_comparison(logger)

        # Задание 1.2: CIFAR-10
        cifar_results = run_cifar10_comparison(logger)

        # Итоговое сравнение
        logger.info("\n" + "="*70)
        logger.info("ИТОГОВОЕ СРАВНЕНИЕ")
        logger.info("="*70)

        # Лучшие модели
        best_mnist, best_mnist_result = get_best_model(mnist_results)
        best_cifar, best_cifar_result = get_best_model(cifar_results)

        logger.info(f"\nMNIST - Лучшая модель: {best_mnist}")
        logger.info(f"  Test Accuracy: {best_mnist_result['best_test_acc']:.2f}%")
        logger.info(f"  Параметров: {best_mnist_result['params']:,}")
        logger.info(f"  Время обучения: {best_mnist_result['training_time']:.2f}s")

        logger.info(f"\nCIFAR-10 - Лучшая модель: {best_cifar}")
        logger.info(f"  Test Accuracy: {best_cifar_result['best_test_acc']:.2f}%")
        logger.info(f"  Параметров: {best_cifar_result['params']:,}")
        logger.info(f"  Время обучения: {best_cifar_result['training_time']:.2f}s")

        # Ключевые выводы
        logger.info("\n" + "="*70)
        logger.info("КЛЮЧЕВЫЕ ВЫВОДЫ")
        logger.info("="*70)
        logger.info("1. Сверточные сети значительно превосходят полносвязные на обоих датасетах")
        logger.info("2. Residual блоки помогают обучать более глубокие сети")
        logger.info("3. Регуляризация критична для CIFAR-10 для борьбы с переобучением")
        logger.info("4. CNN имеют меньше параметров, но лучшее качество за счет использования структуры изображений")
        logger.info("5. На MNIST разница между моделями менее заметна из-за простоты датасета")
        logger.info("6. На CIFAR-10 разница значительно больше, особенно с регуляризацией")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        logger.info("\n" + "="*70)
        logger.info(f"ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
        logger.info(f"Общее время выполнения: {total_time:.2f} секунд ({total_time/60:.2f} минут)")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()