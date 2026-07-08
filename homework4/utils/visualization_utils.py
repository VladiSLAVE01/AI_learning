"""
Утилиты для визуализации результатов
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ==================== НАСТРОЙКА СТИЛЯ ====================

def setup_plot_style():
    """
    Настройка стиля графиков
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10

# ==================== КРИВЫЕ ОБУЧЕНИЯ ====================

def plot_training_curves(history: Dict, title: str = "Training Curves",
                         save_path: str = None, show: bool = True):
    """
    Визуализирует кривые обучения

    Args:
        history: словарь с историей обучения
        title: заголовок графика
        save_path: путь для сохранения
        show: показывать график
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Потери
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(history['test_losses'], label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Точность
    axes[1].plot(history['train_accs'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['test_accs'], label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Добавляем аннотацию с лучшим значением
    if 'best_test_acc' in history:
        best_epoch = history.get('best_epoch', 0)
        best_acc = history.get('best_test_acc', 0)
        axes[1].annotate(f'Best: {best_acc:.2f}% (Epoch {best_epoch})',
                         xy=(best_epoch, best_acc),
                         xytext=(10, 10),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Сохранение
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig

def plot_multiple_training_curves(results: Dict[str, Dict], title: str = "Model Comparison",
                                  save_path: str = None, show: bool = True):
    """
    Визуализирует кривые обучения для нескольких моделей
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (model_name, result) in enumerate(results.items()):
        color = colors[idx]

        # Потери
        axes[0].plot(result['train_losses'], label=f'{model_name} (train)',
                     linestyle='-', color=color, linewidth=2, alpha=0.7)
        axes[0].plot(result['test_losses'], label=f'{model_name} (test)',
                     linestyle='--', color=color, linewidth=2, alpha=0.7)

        # Точность
        axes[1].plot(result['train_accs'], label=f'{model_name} (train)',
                     linestyle='-', color=color, linewidth=2, alpha=0.7)
        axes[1].plot(result['test_accs'], label=f'{model_name} (test)',
                     linestyle='--', color=color, linewidth=2, alpha=0.7)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig

# ==================== CONFUSION MATRIX ====================

def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix",
                         save_path=None, show=True, normalize=False):
    """
    Визуализирует confusion matrix
    """
    setup_plot_style()

    # Вычисляем confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=ax, cbar=True, square=True)

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return cm, fig

# ==================== СРАВНИТЕЛЬНЫЕ ГРАФИКИ ====================

def plot_accuracy_comparison(results: Dict[str, Dict], title="Accuracy Comparison",
                            save_path=None, show=True):
    """
    Визуализирует сравнение точности моделей
    """
    setup_plot_style()

    model_names = list(results.keys())
    train_accs = [results[name]['final_train_acc'] for name in model_names]
    test_accs = [results[name]['final_test_acc'] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy',
                   alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',
                   alpha=0.8, color='coral')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Добавляем значения над столбцами
    for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
        ax.text(i - width/2, train_acc + 0.5, f'{train_acc:.1f}%',
                ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, test_acc + 0.5, f'{test_acc:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig

def plot_time_comparison(results: Dict[str, Dict], title="Training Time Comparison",
                        save_path=None, show=True):
    """
    Визуализирует сравнение времени обучения
    """
    setup_plot_style()

    model_names = list(results.keys())
    train_times = []
    inference_times = []

    for name in model_names:
        train_time = results[name].get('training_time', 0)
        inference_time = results[name].get('inference_time', 0)
        if inference_time == 0 and train_time > 0:
            inference_time = train_time * 0.05
        train_times.append(train_time)
        inference_times.append(inference_time)

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width/2, train_times, width, label='Training Time',
                   alpha=0.8, color='forestgreen')
    bars2 = ax.bar(x + width/2, inference_times, width, label='Inference Time (estimated)',
                   alpha=0.8, color='goldenrod')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for i, (train_time, infer_time) in enumerate(zip(train_times, inference_times)):
        if train_time > 0:
            ax.text(i - width/2, train_time + 0.1, f'{train_time:.1f}s',
                    ha='center', va='bottom', fontsize=9)
        if infer_time > 0:
            ax.text(i + width/2, infer_time + 0.1, f'{infer_time:.3f}s',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig

def plot_parameter_comparison(results: Dict[str, Dict], title="Parameter Comparison",
                             save_path=None, show=True):
    """
    Визуализирует сравнение количества параметров
    """
    setup_plot_style()

    model_names = list(results.keys())
    params = [results[name]['params'] for name in model_names]
    test_accs = [results[name]['final_test_acc'] for name in model_names]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bars = ax1.bar(model_names, params, alpha=0.8, color='steelblue')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Number of Parameters', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (bar, param) in enumerate(zip(bars, params)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{param:,}', ha='center', va='bottom', fontsize=9, rotation=0)

    ax2 = ax1.twinx()
    ax2.plot(model_names, test_accs, 'r-o', linewidth=2, markersize=8, label='Test Accuracy')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.2)

    for i, acc in enumerate(test_accs):
        ax2.text(i, acc + 0.5, f'{acc:.1f}%', ha='center', va='bottom',
                fontsize=9, color='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig

# ==================== ГРАДИЕНТЫ ====================

def plot_gradient_distribution(model, dataloader, device, save_path=None, show=True):
    """
    Визуализирует распределение градиентов
    """
    setup_plot_style()

    model.train()

    # Получаем один батч
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)

    # Forward и backward
    model.zero_grad()
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    grad_data = []
    layer_names = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().cpu().numpy().flatten()
            grad_data.append(grad)
            layer_names.append(name)

    for i in range(min(4, len(grad_data))):
        axes[i].hist(grad_data[i], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_title(f'Layer: {layer_names[i][:30]}', fontsize=10)
        axes[i].set_xlabel('Gradient Value')
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Gradient Distribution by Layer', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig

def plot_gradient_flow(model, dataloader, device, save_path=None, show=True):
    """
    Визуализирует поток градиентов через слои
    """
    setup_plot_style()

    model.train()

    # Получаем один батч
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)

    # Forward и backward
    model.zero_grad()
    output = model(data)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()

    # Собираем нормы градиентов
    grad_norms = []
    layer_names = []

    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            layer_names.append(name)

    if not grad_norms:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No gradients found', ha='center', va='center', fontsize=14)
        ax.set_title('Gradient Flow - No gradients')
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.RdYlGn(np.array(grad_norms) / max(grad_norms) if grad_norms else 0)
    bars = ax.bar(range(len(layer_names)), grad_norms, color=colors, alpha=0.7)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Flow Through Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels([name[:15] for name in layer_names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, norm) in enumerate(zip(bars, grad_norms)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{norm:.4f}', ha='center', va='bottom', fontsize=8)

    if grad_norms:
        mean_norm = np.mean(grad_norms)
        ax.axhline(y=mean_norm, color='red', linestyle='--', alpha=0.5,
                   label=f'Mean: {mean_norm:.4f}')
        ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig

# ==================== ВСЕ ГРАФИКИ СРАВНЕНИЯ ====================

def plot_full_comparison(results: Dict[str, Dict], base_path: str, title_prefix: str = ""):
    """
    Создает все графики для сравнения моделей
    """
    os.makedirs(base_path, exist_ok=True)

    plot_multiple_training_curves(
        results,
        title=f"{title_prefix} Training Curves",
        save_path=f"{base_path}/training_curves.png"
    )

    plot_accuracy_comparison(
        results,
        title=f"{title_prefix} Accuracy Comparison",
        save_path=f"{base_path}/accuracy_comparison.png"
    )

    plot_time_comparison(
        results,
        title=f"{title_prefix} Time Comparison",
        save_path=f"{base_path}/time_comparison.png"
    )

    plot_parameter_comparison(
        results,
        title=f"{title_prefix} Parameter Comparison",
        save_path=f"{base_path}/parameter_comparison.png"
    )

    logger.info(f"All plots saved to: {base_path}")