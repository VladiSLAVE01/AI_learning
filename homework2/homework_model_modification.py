"""Задание 1: Модификация существующих моделей

Расширение базовой линейной и логистической регрессии из regression_basics:

1.1 Линейная регрессия
     L1- и L2-регуляризация (реализованы вручную как добавка к функции потерь);
     ранняя остановка (early stopping) по валидационной ошибке с
     восстановлением лучших весов.

1.2 Логистическая регрессия
    поддержка многоклассовой классификации (softmax + CrossEntropyLoss);
    метрики precision, recall, F1-score и ROC-AUC, реализованные с нуля
    (проверяются на совпадение с scikit-learn в test_homework.py);
    визуализация confusion matrix.

Функции и классы модуля переиспользуются в заданиях 2 и 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    BASE_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    get_device,
    get_logger,
    plot_curves,
    set_seed,
)

logger = get_logger("model_modification")


#  Модели 
class LinearRegressionModel(nn.Module):
    "Линейная регрессия y = Wx + b на базе nn.Linear"

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LogisticRegressionModel(nn.Module):
    "Логистическая (softmax) регрессия для K классов"

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


#  1.1 Регуляризация и ранняя остановка 
def l1_penalty(model: nn.Module) -> torch.Tensor:
    "Сумма модулей весов (без смещений) - штраф L1, ведущий к разреженности"
    return sum(
        param.abs().sum()
        for name, param in model.named_parameters()
        if name.endswith("weight")
    )


def l2_penalty(model: nn.Module) -> torch.Tensor:
    "Сумма квадратов весов (без смещений) - штраф L2"
    return sum(
        param.pow(2).sum()
        for name, param in model.named_parameters()
        if name.endswith("weight")
    )


class EarlyStopping:
    """Ранняя остановка обучения по валидационной метрике.

    Отслеживает лучшее значение loss; если оно не улучшается на min_delta
    в течение patience эпох подряд - сигнализирует об остановке. При
    restore_best=True сохраняет и по запросу возвращает лучшие веса.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False
        self._best_state: Dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        "Обновляет состояние по итогам эпохи, возвращает флаг остановки"
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self._best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model: nn.Module) -> None:
        "Загружает в модель лучшие веса, зафиксированные во время обучения"
        if self.restore_best and self._best_state is not None:
            model.load_state_dict(self._best_state)


@torch.no_grad()
def evaluate_regression(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    "Считает MSE, MAE и R^2 регрессионной модели на переданном загрузчике"
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu())
        targets.append(yb)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum() + 1e-12
    r2 = (1 - ss_res / ss_tot).item()
    return {"mse": mse, "mae": mae, "r2": r2}


def train_linear_regression(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 300,
    lr: float = 0.05,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    patience: int = 20,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict[str, list]:
    "Обучает линейную регрессию с L1/L2-регуляризацией и ранней остановкой"
    device = device or get_device()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience)
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            data_loss = criterion(pred, yb)
            loss = data_loss
            if l1_lambda:
                loss = loss + l1_lambda * l1_penalty(model)
            if l2_lambda:
                loss = loss + l2_lambda * l2_penalty(model)
            loss.backward()
            optimizer.step()
            running += data_loss.item() * xb.size(0)  

        train_loss = running / len(train_loader.dataset)
        val_loss = evaluate_regression(model, val_loader, device)["mse"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            logger.info("Эпоха %3d | train MSE=%.4f | val MSE=%.4f", epoch, train_loss, val_loss)

        if stopper.step(val_loss, model, epoch):
            logger.info(
                "Ранняя остановка на эпохе %d (лучшая эпоха %d, val MSE=%.4f)",
                epoch, stopper.best_epoch, stopper.best_loss,
            )
            break

    stopper.restore(model)
    history["best_epoch"] = stopper.best_epoch or len(history["train_loss"])
    return history


# 1.2 Метрики классификации
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:

    "Матрица ошибок cm[i, j] = число объектов класса i, предсказанных как j"
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(cm: np.ndarray) -> Dict[str, float]:
    """Macro усредненные precision, recall и F1 по матрице ошибок

    Для каждого класса c:
        precision = TP / (TP + FP),  recall = TP / (TP + FN),
        F1 = 2 * precision * recall / (precision + recall)
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    "Ранги значений (1..n) со средним рангом для совпадающих элементов"
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    sorted_values = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _binary_roc_auc(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    "ROC-AUC для бинарной задачи через статистику Манна–Уитни"
    y_true_bin = np.asarray(y_true_bin).astype(int)
    n_pos = int(y_true_bin.sum())
    n_neg = len(y_true_bin) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan") 
    ranks = _rankdata(scores)
    sum_ranks_pos = ranks[y_true_bin == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray, num_classes: int) -> float:
    "ROC-AUC. Для K>2 - one-vs-rest с macro-усреднением"
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if num_classes == 2:
        return _binary_roc_auc((y_true == 1).astype(int), scores[:, 1])
    aucs = [
        _binary_roc_auc((y_true == c).astype(int), scores[:, c])
        for c in range(num_classes)
    ]
    aucs = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


@torch.no_grad()
def evaluate_classification(
    model: nn.Module, loader: DataLoader, num_classes: int, device: torch.device
) -> Dict[str, object]:
    "Полная оценка классификатора: accuracy, precision, recall, F1, ROC-AUC, cm"
    model.eval()
    logits_all, targets_all = [], []
    for xb, yb in loader:
        logits_all.append(model(xb.to(device)).cpu())
        targets_all.append(yb.view(-1))
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all).long().numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    cm = confusion_matrix(targets, preds, num_classes)
    prf = precision_recall_f1(cm)
    return {
        "accuracy": float((preds == targets).mean()),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "roc_auc": roc_auc_score(targets, probs, num_classes),
        "confusion_matrix": cm,
        "targets": targets,
        "preds": preds,
        "probs": probs,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Path | str,
    title: str = "Confusion matrix",
    normalize: bool = False,
) -> Path:
    "Рисует матрицу ошибок в виде теплокарты с числовыми подписями"
    save_path = Path(save_path)
    matrix = cm.astype(float)
    if normalize:
        matrix = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1e-12)

    fig, ax = plt.subplots(figsize=(1.6 * len(class_names) + 2, 1.6 * len(class_names) + 1.5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title)

    threshold = matrix.max() / 2.0
    fmt = ".2f" if normalize else "d"
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j] if normalize else int(cm[i, j])
            ax.text(
                j, i, format(value, fmt),
                ha="center", va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def train_logistic_regression(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    num_classes: int,
    epochs: int = 150,
    lr: float = 0.1,
    optimizer_name: str = "Adam",
    l2_lambda: float = 0.0,
    patience: int = 20,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict[str, list]:
    "Обучает softmax-регрессию с ранней остановкой по валидационному loss"
    device = device or get_device()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)
    stopper = EarlyStopping(patience=patience)
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).long().view(-1)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if l2_lambda:
                loss = loss + l2_lambda * l2_penalty(model)
            loss.backward()
            optimizer.step()
            running += criterion(logits, yb).item() * xb.size(0)

        train_loss = running / len(train_loader.dataset)
        val_metrics = evaluate_classification(model, val_loader, num_classes, device)
        val_loss = _cross_entropy_on_loader(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])

        if verbose and (epoch % 20 == 0 or epoch == 1):
            logger.info(
                "Эпоха %3d | train CE=%.4f | val CE=%.4f | val acc=%.4f",
                epoch, train_loss, val_loss, val_metrics["accuracy"],
            )

        if stopper.step(val_loss, model, epoch):
            logger.info(
                "Ранняя остановка на эпохе %d (лучшая эпоха %d, val CE=%.4f)",
                epoch, stopper.best_epoch, stopper.best_loss,
            )
            break

    stopper.restore(model)
    history["best_epoch"] = stopper.best_epoch or len(history["train_loss"])
    return history


@torch.no_grad()
def _cross_entropy_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    "Средняя кросс-энтропия модели на загрузчике"
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total, n = 0.0, 0
    for xb, yb in loader:
        logits = model(xb.to(device))
        yb = yb.to(device).long().view(-1)
        total += criterion(logits, yb).item()
        n += xb.size(0)
    return total / max(n, 1)


def build_optimizer(name: str, params, lr: float) -> optim.Optimizer:
    "Фабрика оптимизаторов: поддерживает SGD, Adam, RMSprop"
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr)
    if name == "adam":
        return optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    raise ValueError(f"Неизвестный оптимизатор: {name!r}")


#  Загрузчики из тензоров 
def make_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    "Делит выборку на train/val и оборачивает в DataLoader."
    n = X.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_val = int(n * val_ratio)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


#  Демонстрация 
def demo_linear_regression() -> None:
    "1.1 Регуляризация L1/L2 + early stopping"
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler
    from utils import plot_bars

    print("\n1.1 Линейная регрессия: L1/L2-регуляризация и ранняя остановка")

    n_features = 30
    X_np, y_np = make_regression(
        n_samples=80, n_features=n_features, n_informative=10, noise=15.0, random_state=0
    )
    X_np = StandardScaler().fit_transform(X_np)
    y_np = (y_np - y_np.mean()) / y_np.std()
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    train_loader, val_loader = make_loaders(X, y, batch_size=16, val_ratio=0.3)
    device = get_device()

    configs = {
        "Без регуляризации": dict(l1_lambda=0.0, l2_lambda=0.0),
        "L1 (λ=0.01)": dict(l1_lambda=0.01, l2_lambda=0.0),
        "L1 (λ=0.05)": dict(l1_lambda=0.05, l2_lambda=0.0),
        "L2 (λ=0.01)": dict(l1_lambda=0.0, l2_lambda=0.01),
    }
    val_curves: Dict[str, list] = {}
    sparsity: Dict[str, float] = {}
    print("\nСравнение регуляризаций (400 эпох, без ранней остановки):")
    for name, reg in configs.items():
        set_seed(0)
        model = LinearRegressionModel(in_features=n_features)
        history = train_linear_regression(
            model, train_loader, val_loader,
            epochs=400, lr=0.03, patience=10 ** 9, verbose=False, **reg,
        )
        train_mse = evaluate_regression(model, train_loader, device)["mse"]
        val = evaluate_regression(model, val_loader, device)
        weights = model.linear.weight.detach().cpu().numpy().ravel()
        n_near_zero = int((np.abs(weights) < 1e-2).sum())
        print(
            f"  {name:20s} train MSE={train_mse:.4f}  val MSE={val['mse']:.4f}  "
            f"R2={val['r2']:.3f}  |w|_1={np.abs(weights).sum():.2f}  "
            f"почти нулевых весов={n_near_zero}/{n_features}"
        )
        val_curves[name] = history["val_loss"]
        sparsity[name] = n_near_zero

    plot_curves(
        val_curves,
        title="Линейная регрессия: валидационный MSE при разной регуляризации",
        ylabel="Val MSE",
        save_path=PLOTS_DIR / "linreg_regularization.png",
        logy=True,
    )
    plot_bars(
        sparsity,
        title="Разреженность весов: L1 обнуляет больше признаков, чем L2",
        ylabel=f"Число весов |w|<0.01 (из {n_features})",
        save_path=PLOTS_DIR / "linreg_sparsity.png",
    )

    print("\nРанняя остановка (без регуляризации, patience=25):")
    set_seed(0)
    es_model = LinearRegressionModel(in_features=n_features)
    es_history = train_linear_regression(
        es_model, train_loader, val_loader, epochs=400, lr=0.03, patience=25, verbose=False
    )
    print(
        f"  обучение остановлено на эпохе {len(es_history['train_loss'])} из 400, "
        f"восстановлены веса лучшей эпохи {es_history['best_epoch']} "
        f"(val MSE={evaluate_regression(es_model, val_loader, device)['mse']:.4f})"
    )
    plot_curves(
        {"train MSE": es_history["train_loss"], "val MSE": es_history["val_loss"]},
        title="Линейная регрессия: кривые обучения при ранней остановке",
        ylabel="MSE",
        save_path=PLOTS_DIR / "linreg_early_stopping.png",
        logy=True,
    )

    set_seed(0)
    final_model = LinearRegressionModel(in_features=n_features)
    train_linear_regression(
        final_model, train_loader, val_loader, epochs=400, lr=0.03, l2_lambda=0.01, verbose=False
    )
    model_path = MODELS_DIR / "linreg_regularized.pth"
    torch.save(final_model.state_dict(), model_path)
    print(
        f"\nГрафики: linreg_regularization.png, linreg_sparsity.png, linreg_early_stopping.png"
        f"\nМодель сохранена: {model_path.relative_to(BASE_DIR.parent)}"
    )


def demo_logistic_regression() -> None:
    "1.2 Многоклассовая логистическая регрессия: метрики и confusion matrix"
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler

    print("\n1.2 Логистическая регрессия: многоклассовость, метрики, confusion matrix")

    num_classes = 3
    X_np, y_np = make_classification(
        n_samples=600, n_features=8, n_informative=6, n_redundant=1,
        n_classes=num_classes, n_clusters_per_class=1, random_state=42,
    )
    X_np = StandardScaler().fit_transform(X_np)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    train_loader, val_loader = make_loaders(X, y, batch_size=32)

    set_seed(42)
    model = LogisticRegressionModel(in_features=8, num_classes=num_classes)
    history = train_logistic_regression(
        model, train_loader, val_loader, num_classes=num_classes, epochs=150, lr=0.1
    )

    device = get_device()
    metrics = evaluate_classification(model, val_loader, num_classes, device)
    print(
        f"\nМетрики на валидации (macro): accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}"
    )

    cm_path = plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=[f"Класс {i}" for i in range(num_classes)],
        save_path=PLOTS_DIR / "logreg_confusion_matrix.png",
        title="Логистическая регрессия: confusion matrix на валидации",
    )
    loss_path = plot_curves(
        {"train CE": history["train_loss"], "val CE": history["val_loss"]},
        title="Логистическая регрессия: кривые обучения",
        ylabel="Cross-Entropy",
        save_path=PLOTS_DIR / "logreg_training.png",
    )
    print(f"Графики сохранены: {cm_path.name}, {loss_path.name}")

    model_path = MODELS_DIR / "logreg_multiclass.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена: {model_path.relative_to(BASE_DIR.parent)}")


def main() -> None:
    set_seed(42)
    logger.info("Устройство: %s", get_device())
    demo_linear_regression()
    demo_logistic_regression()
    print("\nЗадание 1 выполнено.")


if __name__ == "__main__":
    main()
