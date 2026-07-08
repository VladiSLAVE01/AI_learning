"""
homework_model_modification.py
Задание 1: Модификация существующих моделей

1.1 Линейная регрессия с L1/L2 регуляризацией и Early Stopping
1.2 Логистическая регрессия с поддержкой многоклассовой классификации
    и метриками (precision, recall, F1, ROC-AUC, confusion matrix)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize
import logging
import os
from typing import Dict, List, Tuple, Optional, Any

# Импортируем из utils.py
from utils import (
    make_regression_data,
    make_classification_data,
    RegressionDataset,
    ClassificationDataset,
    mse,
    accuracy,
    log_epoch
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 1.1 ЛИНЕЙНАЯ РЕГРЕССИЯ С РЕГУЛЯРИЗАЦИЕЙ ====================

class LinearRegressionManual:
    """
    Линейная регрессия с L1 и L2 регуляризацией (ручная реализация).

    Использует функции из utils.py:
    - mse() для вычисления ошибки
    - log_epoch() для логирования
    """
    def __init__(self, in_features: int, l1_lambda: float = 0.0, l2_lambda: float = 0.0):
        """
        Args:
            in_features: Количество входных признаков
            l1_lambda: Коэффициент L1 регуляризации (Lasso)
            l2_lambda: Коэффициент L2 регуляризации (Ridge)
        """
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # Для early stopping
        self.best_w = None
        self.best_b = None
        self.best_val_loss = float('inf')

        logger.info(f"LinearRegressionManual: in_features={in_features}, "
                   f"l1={l1_lambda}, l2={l2_lambda}")

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Прямой проход: y = X @ w + b."""
        return X @ self.w + self.b

    def parameters(self):
        """Возвращает параметры модели."""
        return [self.w, self.b]

    def zero_grad(self):
        """Обнуление градиентов."""
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor):
        """
        Обратный проход с вычислением градиентов.
        Включает градиенты от L1 и L2 регуляризации.
        """
        n = X.shape[0]
        error = y_pred - y

        # Градиенты от MSE (используем формулу из utils)
        dw_mse = (X.T @ error) / n
        db_mse = error.mean(0)

        # Градиенты от L1 регуляризации (Lasso)
        if self.l1_lambda > 0:
            dw_l1 = self.l1_lambda * torch.sign(self.w)
        else:
            dw_l1 = 0

        # Градиенты от L2 регуляризации (Ridge)
        if self.l2_lambda > 0:
            dw_l2 = self.l2_lambda * 2 * self.w
        else:
            dw_l2 = 0

        # Суммарные градиенты
        self.dw = dw_mse + dw_l1 + dw_l2
        self.db = db_mse

    def step(self, lr: float):
        """Обновление параметров."""
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Вычисление полной функции потерь с регуляризацией.
        Использует mse() из utils.
        """
        y_pred = self(X)
        mse_loss = mse(y_pred, y)

        # L1 регуляризация
        l1_loss = self.l1_lambda * torch.abs(self.w).sum().item()

        # L2 регуляризация
        l2_loss = self.l2_lambda * (self.w ** 2).sum().item()

        return mse_loss + l1_loss + l2_loss

    def compute_mse(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Вычисление только MSE без регуляризации. Использует mse() из utils."""
        y_pred = self(X)
        return mse(y_pred, y)

    def save_best(self):
        """Сохраняет текущие веса как лучшие."""
        self.best_w = self.w.clone()
        self.best_b = self.b.clone()

    def restore_best(self):
        """Восстанавливает лучшие веса."""
        if self.best_w is not None:
            self.w = self.best_w.clone()
            self.b = self.best_b.clone()
            logger.info("Restored best model weights")

    def save(self, path: str):
        """Сохранение модели."""
        torch.save({
            'w': self.w,
            'b': self.b,
            'best_w': self.best_w,
            'best_b': self.best_b,
            'best_val_loss': self.best_val_loss,
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Загрузка модели."""
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']
        self.best_w = state.get('best_w')
        self.best_b = state.get('best_b')
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        logger.info(f"Model loaded from {path}")


class EarlyStopping:
    """
    Ранняя остановка обучения.

    Отслеживает валидационную ошибку и останавливает обучение,
    если ошибка не улучшается в течение заданного количества эпох.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            patience: Количество эпох без улучшения до остановки
            min_delta: Минимальное изменение для считания улучшением
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

        logger.info(f"EarlyStopping: patience={patience}, min_delta={min_delta}")

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """
        Проверка условия остановки.

        Args:
            val_loss: Текущее значение валидационной потери
            epoch: Номер текущей эпохи

        Returns:
            True если нужно остановить обучение
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping! Best loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
                return True
        return False


def train_linear_regression(
    model: LinearRegressionManual,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.1,
    use_regularization: bool = True,
    early_stopping: Optional[EarlyStopping] = None,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Обучение линейной регрессии с регуляризацией и ранней остановкой.

    Использует:
    - log_epoch() из utils для логирования
    - mse() из utils для вычисления ошибки

    Args:
        model: Модель для обучения
        train_loader: Загрузчик тренировочных данных
        val_loader: Загрузчик валидационных данных
        epochs: Количество эпох
        lr: Скорость обучения
        use_regularization: Использовать регуляризацию
        early_stopping: Объект ранней остановки
        save_path: Путь для сохранения модели
        verbose: Выводить логи

    Returns:
        Словарь с историей обучения
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': []
    }

    logger.info(f"Starting training for {epochs} epochs...")
    if use_regularization:
        logger.info(f"Using regularization: L1={model.l1_lambda}, L2={model.l2_lambda}")

    for epoch in range(1, epochs + 1):
        # Training phase
        train_loss = 0.0
        train_mse_sum = 0.0
        n_batches = 0

        for X, y in train_loader:
            y_pred = model(X)

            # Вычисляем MSE (используем mse из utils)
            mse_loss = mse(y_pred, y)

            # Вычисляем полную потерю с регуляризацией
            if use_regularization:
                l1_loss = model.l1_lambda * torch.abs(model.w).sum().item()
                l2_loss = model.l2_lambda * (model.w ** 2).sum().item()
                total_loss = mse_loss + l1_loss + l2_loss
            else:
                total_loss = mse_loss

            # Backward и шаг оптимизации
            model.zero_grad()
            model.backward(X, y, y_pred)
            model.step(lr)

            train_loss += total_loss
            train_mse_sum += mse_loss
            n_batches += 1

        avg_train_loss = train_loss / n_batches
        avg_train_mse = train_mse_sum / n_batches

        # Validation phase
        val_loss = 0.0
        val_mse_sum = 0.0
        n_val_batches = 0

        for X, y in val_loader:
            y_pred = model(X)
            mse_loss = mse(y_pred, y)

            if use_regularization:
                l1_loss = model.l1_lambda * torch.abs(model.w).sum().item()
                l2_loss = model.l2_lambda * (model.w ** 2).sum().item()
                total_loss = mse_loss + l1_loss + l2_loss
            else:
                total_loss = mse_loss

            val_loss += total_loss
            val_mse_sum += mse_loss
            n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches
        avg_val_mse = val_mse_sum / n_val_batches

        # Сохранение истории
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_mse'].append(avg_train_mse)
        history['val_mse'].append(avg_val_mse)

        # Сохранение лучшей модели
        if avg_val_loss < model.best_val_loss:
            model.best_val_loss = avg_val_loss
            model.save_best()
            if save_path:
                model.save(save_path)

        # Early stopping
        if early_stopping:
            if early_stopping(avg_val_loss, epoch):
                model.restore_best()
                break

        # Логирование (используем log_epoch из utils)
        if epoch % 10 == 0 and verbose:
            log_epoch(epoch, avg_train_loss,
                     val_loss=avg_val_loss,
                     train_mse=avg_train_mse,
                     val_mse=avg_val_mse)

    logger.info("Training completed!")
    return history


# ==================== 1.2 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (МНОГОКЛАССОВАЯ) ====================

class LogisticRegressionMulticlass(nn.Module):
    """
    Логистическая регрессия для многоклассовой классификации (PyTorch версия).

    Поддерживает:
    - Бинарную классификацию (2 класса)
    - Многоклассовую классификацию (> 2 классов)

    Использует:
    - accuracy() из utils для вычисления точности
    """
    def __init__(self, in_features: int, num_classes: int = 2, l2_lambda: float = 0.0):
        """
        Args:
            in_features: Количество входных признаков
            num_classes: Количество классов
            l2_lambda: Коэффициент L2 регуляризации
        """
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.l2_lambda = l2_lambda

        logger.info(f"LogisticRegressionMulticlass: in_features={in_features}, "
                   f"num_classes={num_classes}, l2={l2_lambda}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход. Возвращает логиты."""
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает вероятности классов."""
        logits = self.forward(x)
        if self.num_classes == 2:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает предсказанные классы."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def regularization_loss(self) -> torch.Tensor:
        """Вычисление L2 регуляризационного штрафа."""
        return self.l2_lambda * sum((p ** 2).sum() for p in self.parameters())


class LogisticRegressionManualMulticlass:
    """
    Логистическая регрессия для многоклассовой классификации (ручная реализация).

    Поддерживает:
    - Бинарную классификацию (2 класса)
    - Многоклассовую классификацию (> 2 классов)

    Использует:
    - accuracy() из utils для вычисления точности
    """
    def __init__(self, in_features: int, num_classes: int = 2, l2_lambda: float = 0.0):
        """
        Args:
            in_features: Количество входных признаков
            num_classes: Количество классов
            l2_lambda: Коэффициент L2 регуляризации
        """
        self.W = torch.randn(in_features, num_classes, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(num_classes, dtype=torch.float32, requires_grad=False)
        self.num_classes = num_classes
        self.l2_lambda = l2_lambda

        # Для early stopping
        self.best_W = None
        self.best_b = None
        self.best_val_loss = float('inf')

        logger.info(f"LogisticRegressionManualMulticlass: in_features={in_features}, "
                   f"num_classes={num_classes}, l2={l2_lambda}")

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """Прямой проход. Возвращает вероятности."""
        logits = X @ self.W + self.b
        if self.num_classes == 2:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=1)

    def logits(self, X: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты."""
        return X @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)

    def backward(self, X: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor):
        """
        Обратный проход с вычислением градиентов.

        Args:
            X: Входные данные (batch_size, in_features)
            y: Целевые метки (batch_size, 1)
            y_pred: Предсказания (batch_size, num_classes)
        """
        n = X.shape[0]

        if self.num_classes == 2:
            # Бинарная классификация
            y = y.squeeze()
            error = y_pred - y
            self.dW = (X.T @ error) / n
            self.db = error.mean(0)
        else:
            # Многоклассовая классификация
            y_one_hot = torch.zeros(n, self.num_classes)
            y_one_hot.scatter_(1, y.long(), 1)
            error = y_pred - y_one_hot
            self.dW = (X.T @ error) / n
            self.db = error.mean(0)

        # L2 регуляризация
        if self.l2_lambda > 0:
            self.dW += self.l2_lambda * 2 * self.W

    def step(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Предсказание классов."""
        logits = X @ self.W + self.b
        return torch.argmax(logits, dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Возвращает вероятности классов."""
        logits = X @ self.W + self.b
        if self.num_classes == 2:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=1)

    def save_best(self):
        self.best_W = self.W.clone()
        self.best_b = self.b.clone()

    def restore_best(self):
        if self.best_W is not None:
            self.W = self.best_W.clone()
            self.b = self.best_b.clone()

    def save(self, path: str):
        torch.save({
            'W': self.W,
            'b': self.b,
            'best_W': self.best_W,
            'best_b': self.best_b,
            'best_val_loss': self.best_val_loss,
            'num_classes': self.num_classes,
            'l2_lambda': self.l2_lambda
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        state = torch.load(path)
        self.W = state['W']
        self.b = state['b']
        self.best_W = state.get('best_W')
        self.best_b = state.get('best_b')
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        self.num_classes = state['num_classes']
        self.l2_lambda = state.get('l2_lambda', 0.0)


# ==================== МЕТРИКИ ДЛЯ КЛАССИФИКАЦИИ ====================

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Вычисление полного набора метрик классификации.

    Использует функции из sklearn:
    - accuracy_score
    - precision_score
    - recall_score
    - f1_score
    - roc_auc_score

    Args:
        y_true: Истинные метки (n_samples,)
        y_pred: Предсказанные метки (n_samples,)
        y_prob: Вероятности (n_samples, n_classes) для ROC-AUC

    Returns:
        Словарь с метриками: accuracy, precision, recall, f1, roc_auc
    """
    metrics = {}

    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1 (weighted average для многоклассовой)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # ROC-AUC
    if y_prob is not None:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            # Бинарная классификация
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob_class1 = y_prob[:, 1]
            else:
                y_prob_class1 = y_prob.flatten()
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob_class1)
            except ValueError:
                metrics['roc_auc'] = 0.5
        else:
            # Многоклассовая
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
            try:
                metrics['roc_auc'] = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')
            except ValueError:
                metrics['roc_auc'] = 0.5

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
):
    """
    Визуализация матрицы неточностей.

    Использует:
    - confusion_matrix из sklearn.metrics
    - seaborn для тепловой карты
    - matplotlib для отображения
    """
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()
    plt.close()


def plot_training_history(history: Dict[str, List], title: str = "Training History",
                         save_path: Optional[str] = None):
    """
    Визуализация истории обучения.

    Args:
        history: Словарь с историей
        title: Заголовок графика
        save_path: Путь для сохранения
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # График потерь
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # График метрик (если есть)
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    has_metrics = any(m in history for m in metrics_to_plot)

    if has_metrics:
        for metric in metrics_to_plot:
            if metric in history:
                axes[1].plot(history[metric], label=metric.capitalize(), linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Value')
        axes[1].set_title('Metrics', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Если метрик нет, показываем MSE
        if 'train_mse' in history and 'val_mse' in history:
            axes[1].plot(history['train_mse'], label='Train MSE', linewidth=2)
            axes[1].plot(history['val_mse'], label='Validation MSE', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MSE')
            axes[1].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    plt.show()
    plt.close()


# ==================== ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ ====================

def train_logistic_regression_torch(
    model: LogisticRegressionMulticlass,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.01,
    use_regularization: bool = True,
    early_stopping: Optional[EarlyStopping] = None,
    save_path: Optional[str] = None
) -> Tuple[Dict[str, List], Dict[str, float]]:
    """
    Обучение логистической регрессии (PyTorch версия) с метриками.

    Использует:
    - accuracy() из utils для вычисления точности
    - log_epoch() из utils для логирования
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    best_val_loss = float('inf')
    best_state_dict = None
    final_metrics = {}

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            y_target = y.squeeze().long()

            loss = criterion(logits, y_target)
            if use_regularization and hasattr(model, 'regularization_loss'):
                loss += model.regularization_loss()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X)
                y_target = y.squeeze().long()

                loss = criterion(logits, y_target)
                if use_regularization and hasattr(model, 'regularization_loss'):
                    loss += model.regularization_loss()
                val_loss += loss.item()

                # Сохранение предсказаний
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_target.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        val_loss /= len(val_loader)

        # Вычисление метрик
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.vstack(all_probs) if all_probs else None

        metrics = calculate_classification_metrics(all_labels, all_preds, all_probs)

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(metrics['accuracy'])
        history['precision'].append(metrics['precision'])
        history['recall'].append(metrics['recall'])
        history['f1'].append(metrics['f1'])

        # Логирование (используем log_epoch из utils)
        if epoch % 10 == 0:
            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            for k, v in metrics.items():
                log_msg += f", {k}={v:.4f}"
            logger.info(log_msg)

        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict().copy()
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state_dict, save_path)

        # Early stopping
        if early_stopping and early_stopping(val_loss, epoch):
            if best_state_dict is not None:
                model.load_state_dict(best_state_dict)
                logger.info("Restored best model state")
            break

    # Финальные метрики
    final_metrics = calculate_classification_metrics(all_labels, all_preds, all_probs)
    logger.info("Final metrics:")
    for k, v in final_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return history, final_metrics


# ==================== ДЕМОНСТРАЦИЯ ====================

def demo_linear_regression():
    """Демонстрация линейной регрессии с регуляризацией."""
    print("1.1 ЛИНЕЙНАЯ РЕГРЕССИЯ С РЕГУЛЯРИЗАЦИЕЙ")

    # Генерируем данные (используем make_regression_data из utils)
    X, y = make_regression_data(n=500, source='random')

    # Создаём датасет (используем RegressionDataset из utils)
    dataset = RegressionDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей train: {len(train_loader)}')
    print(f'Количество батчей val: {len(val_loader)}')

    # Создаем модель с регуляризацией
    model = LinearRegressionManual(
        in_features=X.shape[1],
        l1_lambda=0.001,
        l2_lambda=0.001
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=1e-5)

    # Обучение
    history = train_linear_regression(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.1,
        use_regularization=True,
        early_stopping=early_stopping,
        save_path='models/linreg_manual_with_reg.pth'
    )

    # Визуализация
    plot_training_history(history, title="Linear Regression with L1/L2 Regularization",
                         save_path='plots/linreg_history.png')

    return history


def demo_logistic_regression():
    """Демонстрация логистической регрессии с метриками."""
    print("1.2 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (МНОГОКЛАССОВАЯ)")

    X, y = make_classification_data(n=800, source='random')

    from sklearn.datasets import make_classification as sk_make

    X_np, y_np = sk_make(
	    n_samples=800,
	    n_features=5,
	    n_informative=3,
	    n_redundant=1,
	    n_classes=3,
	    n_clusters_per_class=1,
	    random_state=42
    )

    X = torch.FloatTensor(X_np)
    y = torch.FloatTensor(y_np).unsqueeze(1)

    # Создаём датасет (используем ClassificationDataset из utils)
    dataset = ClassificationDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество классов: {len(torch.unique(y))}')
    print(f'Количество батчей train: {len(train_loader)}')
    print(f'Количество батчей val: {len(val_loader)}')

    # PyTorch версия
    print("\n--- PyTorch версия ---")
    model_torch = LogisticRegressionMulticlass(
        in_features=X.shape[1],
        num_classes=3,
        l2_lambda=0.01
    )

    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

    history_torch, metrics_torch = train_logistic_regression_torch(
        model=model_torch,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=0.01,
        use_regularization=True,
        early_stopping=early_stopping,
        save_path='models/logreg_torch_multiclass.pth'
    )

    # Визуализация
    plot_training_history(history_torch, title="Logistic Regression (PyTorch)",
                         save_path='plots/logreg_torch_history.png')

    # Визуализация матрицы неточностей
    model_torch.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in val_loader:
            preds = model_torch.predict(X)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.squeeze().long().cpu().numpy())

    plot_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        title="Confusion Matrix (PyTorch)",
        save_path='plots/confusion_matrix_torch.png'
    )

    print("\nИтоговые метрики (PyTorch):")
    for k, v in metrics_torch.items():
        print(f"  {k}: {v:.4f}")

    return history_torch, metrics_torch


def main():
    """Основная функция для демонстрации."""
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # Демонстрация линейной регрессии
    demo_linear_regression()

    # Демонстрация логистической регрессии
    demo_logistic_regression()

    print("\n" + "="*60)
    print("ВСЕ ДЕМОНСТРАЦИИ ЗАВЕРШЕНЫ УСПЕШНО!")
    print("="*60)


if __name__ == "__main__":
    main()