"""Задание 3: Эксперименты и анализ

3.1 Исследование гиперпараметров на реальном датасете (insurance.csv):
    скорость обучения (learning rate);
    размер батча;
    оптимизатор (SGD, Adam, RMSprop);
    результаты сводятся в таблицы и графики

3.2 Feature Engineering:
    полиномиальные признаки;
    попарные взаимодействия признаков;
    статистические признакиж
    Качество сравнивается с базовой моделью.

Функции обучения и оценки переиспользуются из homework_model_modification.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import torch

from homework_datasets import load_csv_splits
from homework_model_modification import (
    LinearRegressionModel,
    build_optimizer,
    evaluate_regression,
)
from utils import DATA_DIR, PLOTS_DIR, get_device, get_logger, plot_bars, plot_curves, set_seed

logger = get_logger("experiments")

# Гиперпараметры по умолчанию для базовой конфигурации экспериментов 3.1
DEFAULT_EPOCHS = 200
DEFAULT_LR = 0.05
DEFAULT_BATCH = 32
DEFAULT_OPTIMIZER = "sgd"


#  Обучение по «сырым» тензорам
def train_linear_raw(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    optimizer_name: str = DEFAULT_OPTIMIZER,
    seed: int = 42,
) -> Tuple[LinearRegressionModel, List[float]]:
    "Обучает линейную регрессию, возвращает модель и кривую val MSE по эпохам"
    from torch.utils.data import DataLoader, TensorDataset

    device = get_device()
    set_seed(seed)
    model = LinearRegressionModel(in_features=X_train.shape[1]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = build_optimizer(optimizer_name, model.parameters(), lr)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256, shuffle=False)

    val_curve: List[float] = []
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        val_curve.append(evaluate_regression(model, val_loader, device)["mse"])
    return model, val_curve


def _insurance_tensors():
    "Готовит тензоры insurance.csv и индексы числовых/бинарных признаков"
    train_ds, test_ds = load_csv_splits(
        DATA_DIR / "insurance.csv",
        target_column="charges",
        task="regression",
        numeric_cols=["age", "bmi", "children"],
        binary_cols=["sex", "smoker"],
        categorical_cols=["region"],
        seed=42,
    )
    numeric_idx = [0, 1, 2]  # age, bmi, children
    binary_idx = [3, 4]  # sex, smoker
    return train_ds.X, train_ds.y, test_ds.X, test_ds.y, numeric_idx, binary_idx


#  3.1 Исследование гиперпараметров 
def experiment_learning_rates(
    X_train, y_train, X_val, y_val, rates: Sequence[float] = (0.001, 0.01, 0.05, 0.1, 0.5)
) -> Dict[float, float]:
    "Сравнивает скорость обучения по финальному val MSE и кривым сходимости"
    print("\n 3.1a Скорость обучения (learning rate) ")
    curves, finals = {}, {}
    for lr in rates:
        _, curve = train_linear_raw(X_train, y_train, X_val, y_val, lr=lr)
        curve = [min(c, 10.0) for c in curve]  
        curves[f"lr={lr}"] = curve
        finals[str(lr)] = curve[-1]
        print(f"  lr={lr:<6} финальный val MSE={curve[-1]:.4f}")
    plot_curves(
        curves, title="Insurance: сходимость при разной скорости обучения",
        ylabel="Val MSE", save_path=PLOTS_DIR / "exp_learning_rate.png", logy=True,
    )
    return finals


def experiment_batch_sizes(
    X_train, y_train, X_val, y_val, sizes: Sequence[int] = (8, 16, 32, 64, 128)
) -> Dict[int, float]:
    "Сравнивает размеры батча по финальному val MSE и времени обучения"
    print("\n 3.1b Размер батча ")
    finals, times = {}, {}
    for bs in sizes:
        start = time.perf_counter()
        _, curve = train_linear_raw(X_train, y_train, X_val, y_val, batch_size=bs)
        elapsed = time.perf_counter() - start
        finals[str(bs)] = curve[-1]
        times[str(bs)] = elapsed
        print(f"  batch={bs:<4} финальный val MSE={curve[-1]:.4f}  время={elapsed:.2f}с")
    plot_bars(
        finals, title="Insurance: финальный val MSE при разном размере батча",
        ylabel="Val MSE", save_path=PLOTS_DIR / "exp_batch_size.png", xlabel="Размер батча",
    )
    plot_bars(
        times, title="Insurance: время обучения при разном размере батча",
        ylabel="Секунды", save_path=PLOTS_DIR / "exp_batch_time.png", xlabel="Размер батча",
    )
    return finals


def experiment_optimizers(
    X_train, y_train, X_val, y_val, names: Sequence[str] = ("sgd", "adam", "rmsprop")
) -> Dict[str, float]:
    "Сравнивает оптимизаторы SGD, Adam, RMSprop по сходимости"
    print("\n 3.1c Оптимизаторы ")
    curves, finals = {}, {}
    for name in names:
        _, curve = train_linear_raw(X_train, y_train, X_val, y_val, optimizer_name=name, lr=0.01)
        curves[name.upper()] = curve
        finals[name] = curve[-1]
        print(f"  {name:<8} финальный val MSE={curve[-1]:.4f}")
    plot_curves(
        curves, title="Insurance: сходимость разных оптимизаторов при lr=0.01",
        ylabel="Val MSE", save_path=PLOTS_DIR / "exp_optimizers.png", logy=True,
    )
    return finals


def run_hyperparameter_study() -> None:
    "Полное исследование гиперпараметров 3.1 на insurance.csv"
    print("\n3.1 Исследование гиперпараметров (insurance.csv)")
    X_train_full, y_train_full, _, _, _, _ = _insurance_tensors()
    (X_tr, y_tr), (X_val, y_val) = _split_tensors(X_train_full, y_train_full, val_ratio=0.2)
    experiment_learning_rates(X_tr, y_tr, X_val, y_val)
    experiment_batch_sizes(X_tr, y_tr, X_val, y_val)
    experiment_optimizers(X_tr, y_tr, X_val, y_val)
    print(
        "\nГрафики: exp_learning_rate.png, exp_batch_size.png, exp_batch_time.png, exp_optimizers.png"
    )


def _split_tensors(X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.2, seed: int = 42):
    "Детерминированно делит тензоры на train/val части"
    n = X.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    n_val = int(n * val_ratio)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])


#  3.2 Feature Engineering 
def add_polynomial_features(X: torch.Tensor, indices: Sequence[int], degree: int = 2) -> torch.Tensor:
    "Возвращает степени 2..degree выбранных столбцов indices"
    feats = [X[:, indices] ** d for d in range(2, degree + 1)]
    return torch.cat(feats, dim=1) if feats else X.new_empty((X.shape[0], 0))


def add_interaction_features(X: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    "Возвращает попарные произведения выбранных столбцов indices"
    cols = []
    idx = list(indices)
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            cols.append((X[:, idx[i]] * X[:, idx[j]]).unsqueeze(1))
    return torch.cat(cols, dim=1) if cols else X.new_empty((X.shape[0], 0))


def add_statistical_features(X: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    "Возвращает построчные статистики - среднее, дисперсию, min, max - по indices"
    sub = X[:, indices]
    stats = torch.stack(
        [sub.mean(dim=1), sub.var(dim=1, unbiased=False), sub.amin(dim=1), sub.amax(dim=1)],
        dim=1,
    )
    return stats


def build_engineered_features(
    X: torch.Tensor,
    numeric_idx: Sequence[int],
    interaction_idx: Sequence[int] | None = None,
) -> torch.Tensor:
    "Собирает расширенную матрицу признаков: исходные + poly + interactions + stats"
    if interaction_idx is None:
        interaction_idx = numeric_idx
    return torch.cat(
        [
            X,
            add_polynomial_features(X, numeric_idx, degree=2),
            add_interaction_features(X, interaction_idx),
            add_statistical_features(X, numeric_idx),
        ],
        dim=1,
    )


def _standardize(train: torch.Tensor, *others: torch.Tensor) -> List[torch.Tensor]:
    "Стандартизует тензоры по статистикам train (mean/std)"
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
    return [(t - mean) / std for t in (train, *others)]


def experiment_feature_engineering() -> Dict[str, Dict[str, float]]:
    "3.2 Сравнение базовой модели и модели с инженерными признаками"
    print("\n3.2 Feature Engineering (insurance.csv)")
    X_train, y_train, X_test, y_test, numeric_idx, binary_idx = _insurance_tensors()
    interaction_idx = numeric_idx + binary_idx 

    device = get_device()
    results: Dict[str, Dict[str, float]] = {}

    def evaluate_variant(name: str, X_tr: torch.Tensor, X_te: torch.Tensor) -> None:
        (Xtr, ytr), (Xval, yval) = _split_tensors(X_tr, y_train, val_ratio=0.2)
        model, _ = train_linear_raw(
            Xtr, ytr, Xval, yval, epochs=400, lr=0.03, batch_size=32, optimizer_name="adam"
        )
        from torch.utils.data import DataLoader, TensorDataset

        test_loader = DataLoader(TensorDataset(X_te, y_test), batch_size=256, shuffle=False)
        metrics = evaluate_regression(model, test_loader, device)
        results[name] = {"mse": metrics["mse"], "r2": metrics["r2"], "n_features": X_tr.shape[1]}
        print(
            f"  {name:20s} признаков={X_tr.shape[1]:<3} test MSE={metrics['mse']:.4f}  "
            f"R2={metrics['r2']:.4f}"
        )

    evaluate_variant("Базовая", X_train, X_test)

    # Расширенные признаки со стандартизацией по train-статистикам
    X_train_eng = build_engineered_features(X_train, numeric_idx, interaction_idx)
    X_test_eng = build_engineered_features(X_test, numeric_idx, interaction_idx)
    X_train_eng, X_test_eng = _standardize(X_train_eng, X_test_eng)
    evaluate_variant("Feature engineering", X_train_eng, X_test_eng)

    improvement = results["Feature engineering"]["r2"] - results["Базовая"]["r2"]
    print(f"\nПрирост R2 от feature engineering: {improvement:+.4f}")

    plot_bars(
        {k: v["r2"] for k, v in results.items()},
        title="Feature engineering против базовой модели по R2 на тесте",
        ylabel="R2 на тесте", save_path=PLOTS_DIR / "exp_feature_engineering.png",
    )
    print("График: exp_feature_engineering.png")
    return results


def main() -> None:
    set_seed(42)
    logger.info("Устройство: %s", get_device())
    run_hyperparameter_study()
    experiment_feature_engineering()
    print("\nЗадание 3 выполнено")


if __name__ == "__main__":
    main()
