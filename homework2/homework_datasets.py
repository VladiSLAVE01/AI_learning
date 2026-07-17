"""Задание 2: Работа с датасетами

2.1 Кастомный класс CSVDataset:
    загрузка данных из CSV-файла
    предобработка: нормализация числовых признаков, кодирование
    категориальных и бинарных столбцов, заполнение пропусков
    автоматическое определение типов столбцов либо явное указание
    согласованная предобработка train/test.

2.2 Эксперименты на реальных CSV-датасетах:
    регрессия - data/insurance.csv
    бинарная классификация - data/titanic.csv.

Модели и функции обучения берутся из homework_model_modification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from homework_model_modification import (
    LinearRegressionModel,
    LogisticRegressionModel,
    evaluate_classification,
    evaluate_regression,
    make_loaders,
    plot_confusion_matrix,
    train_linear_regression,
    train_logistic_regression,
)
from utils import DATA_DIR, MODELS_DIR, PLOTS_DIR, get_device, get_logger, plot_curves, set_seed

logger = get_logger("datasets")


#  2.1 Кастомный Dataset для CSV 
@dataclass
class FitState:

    numeric_cols: List[str] = field(default_factory=list)
    binary_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    numeric_median: Dict[str, float] = field(default_factory=dict)
    numeric_mean: Dict[str, float] = field(default_factory=dict)
    numeric_std: Dict[str, float] = field(default_factory=dict)
    binary_map: Dict[str, dict] = field(default_factory=dict)
    categorical_mode: Dict[str, object] = field(default_factory=dict)
    categorical_levels: Dict[str, List[object]] = field(default_factory=dict)
    target_classes: Optional[List[object]] = None
    target_mean: float = 0.0
    target_std: float = 1.0
    feature_names: List[str] = field(default_factory=list)


class CSVDataset(Dataset):
    """Универсальный датасет для табличных CSV-данных.

    Признаки автоматически делятся на числовые, бинарные и категориальные, либо типы задаются явно. 
    Числовые стандартизуются, бинарные приводятся к
    {0, 1}, категориальные разворачиваются в one-hot. Пропуски заполняются
    медианой или модой.
    """

    def __init__(
        self,
        data: "str | Path | pd.DataFrame",
        target_column: str,
        task: str = "regression",
        *,
        numeric_cols: Optional[Sequence[str]] = None,
        binary_cols: Optional[Sequence[str]] = None,
        categorical_cols: Optional[Sequence[str]] = None,
        drop_cols: Optional[Sequence[str]] = None,
        normalize: bool = True,
        standardize_target: bool = True,
        reference: "Optional[CSVDataset]" = None,
    ):
        if task not in ("regression", "classification"):
            raise ValueError("task должен быть 'regression' или 'classification'")
        self.task = task
        self.target_column = target_column
        self.normalize = normalize
        self.standardize_target = standardize_target and task == "regression"

        df = pd.read_csv(data) if not isinstance(data, pd.DataFrame) else data.copy()
        if target_column not in df.columns:
            raise KeyError(f"Целевой столбец '{target_column}' не найден в CSV")
        if drop_cols:
            df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        if reference is None:
            self.state = self._fit(df, numeric_cols, binary_cols, categorical_cols)
        else:
            self.state = reference.state  

        self.X, self.y = self._transform(df)

    def _fit(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[Sequence[str]],
        binary_cols: Optional[Sequence[str]],
        categorical_cols: Optional[Sequence[str]],
    ) -> FitState:
        feature_df = df.drop(columns=[self.target_column])

        if numeric_cols is None and binary_cols is None and categorical_cols is None:
            numeric_cols, binary_cols, categorical_cols = self._infer_types(feature_df)

        state = FitState(
            numeric_cols=list(numeric_cols or []),
            binary_cols=list(binary_cols or []),
            categorical_cols=list(categorical_cols or []),
        )

        # Числовые: медиана + среднее/СКО 
        for col in state.numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            median = float(values.median())
            filled = values.fillna(median)
            state.numeric_median[col] = median
            state.numeric_mean[col] = float(filled.mean())
            std = float(filled.std(ddof=0))
            state.numeric_std[col] = std if std > 1e-8 else 1.0

        # Бинарные: два уникальных значения - {0, 1}, отсортированы по возрастанию
        for col in state.binary_cols:
            uniques = sorted(df[col].dropna().unique().tolist(), key=str)
            state.binary_map[col] = {val: idx for idx, val in enumerate(uniques[:2])}
            state.categorical_mode[col] = df[col].mode(dropna=True).iloc[0]

        # Категориальные: мода + список уровней
        for col in state.categorical_cols:
            state.categorical_mode[col] = df[col].mode(dropna=True).iloc[0]
            levels = sorted(df[col].dropna().unique().tolist(), key=str)
            state.categorical_levels[col] = levels

        # Цель
        if self.task == "classification":
            state.target_classes = sorted(df[self.target_column].dropna().unique().tolist(), key=str)
        elif self.standardize_target:
            target = pd.to_numeric(df[self.target_column], errors="coerce")
            state.target_mean = float(target.mean())
            std = float(target.std(ddof=0))
            state.target_std = std if std > 1e-8 else 1.0

        state.feature_names = self._build_feature_names(state)
        return state

    @staticmethod
    def _infer_types(feature_df: pd.DataFrame):
        "Эвристически делит столбцы на числовые / бинарные / категориальные."
        numeric_cols, binary_cols, categorical_cols = [], [], []
        for col in feature_df.columns:
            series = feature_df[col]
            n_unique = series.dropna().nunique()
            if n_unique <= 2:
                binary_cols.append(col)
            elif pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        return numeric_cols, binary_cols, categorical_cols

    @staticmethod
    def _build_feature_names(state: FitState) -> List[str]:
        names = list(state.numeric_cols) + list(state.binary_cols)
        for col in state.categorical_cols:
            names += [f"{col}={level}" for level in state.categorical_levels[col]]
        return names

    def _transform(self, df: pd.DataFrame):
        state = self.state
        columns = []

        for col in state.numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce").fillna(state.numeric_median[col])
            arr = values.to_numpy(dtype=np.float32)
            if self.normalize:
                arr = (arr - state.numeric_mean[col]) / state.numeric_std[col]
            columns.append(arr.reshape(-1, 1))

        for col in state.binary_cols:
            mapping = state.binary_map[col]
            filled = df[col].fillna(state.categorical_mode[col])
            arr = filled.map(mapping).fillna(0).to_numpy(dtype=np.float32)
            columns.append(arr.reshape(-1, 1))

        for col in state.categorical_cols:
            filled = df[col].fillna(state.categorical_mode[col])
            for level in state.categorical_levels[col]:
                arr = (filled == level).to_numpy(dtype=np.float32)
                columns.append(arr.reshape(-1, 1))

        X = np.hstack(columns).astype(np.float32) if columns else np.empty((len(df), 0), np.float32)
        X = torch.from_numpy(X)

        target = df[self.target_column]
        if self.task == "classification":
            class_to_idx = {c: i for i, c in enumerate(state.target_classes)}
            y = torch.tensor(target.map(class_to_idx).to_numpy(dtype=np.int64), dtype=torch.long)
        else:
            y_np = pd.to_numeric(target, errors="coerce").to_numpy(dtype=np.float32)
            if self.standardize_target:
                y_np = (y_np - state.target_mean) / state.target_std
            y = torch.from_numpy(y_np).unsqueeze(1)
        return X, y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def feature_names(self) -> List[str]:
        return self.state.feature_names

    @property
    def num_classes(self) -> int:
        if self.task != "classification":
            raise AttributeError("num_classes определён только для классификации")
        return len(self.state.target_classes)


def load_csv_splits(
    path: "str | Path",
    target_column: str,
    task: str,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    **col_kwargs,
):
    "Загружает CSV и возвращает согласованные train/test CSVDataset."
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    stratify = df[target_column] if task == "classification" else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify
    )
    train_ds = CSVDataset(train_df.reset_index(drop=True), target_column, task, **col_kwargs)
    test_ds = CSVDataset(
        test_df.reset_index(drop=True), target_column, task, reference=train_ds, **col_kwargs
    )
    return train_ds, test_ds


#  2.2 Эксперименты на реальных датасетах 
def _plot_regression_fit(y_true, y_pred, save_path: Path, title: str) -> None:
    "Диаграмма рассеяния «предсказание против истины» для регрессии."
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=18, color="#4C72B0")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", label="идеальное совпадение")
    plt.xlabel("Истинное значение, станд.")
    plt.ylabel("Предсказание")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def run_regression_experiment() -> Dict[str, float]:
    "Линейная регрессия на insurance.csv"
    print("\n2.2 Регрессия: стоимость медицинской страховки (insurance.csv)")

    train_ds, test_ds = load_csv_splits(
        DATA_DIR / "insurance.csv",
        target_column="charges",
        task="regression",
        numeric_cols=["age", "bmi", "children"],
        binary_cols=["sex", "smoker"],
        categorical_cols=["region"],
        seed=42,
    )
    logger.info(
        "insurance: train=%d, test=%d, признаков после кодирования=%d",
        len(train_ds), len(test_ds), train_ds.n_features,
    )
    print(f"Признаки: {train_ds.feature_names}")

    train_loader, val_loader = make_loaders(train_ds.X, train_ds.y, batch_size=32, val_ratio=0.2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    set_seed(42)
    model = LinearRegressionModel(in_features=train_ds.n_features)
    history = train_linear_regression(
        model, train_loader, val_loader,
        epochs=400, lr=0.05, l2_lambda=0.001, patience=30, verbose=False,
    )

    device = get_device()
    test_metrics = evaluate_regression(model, test_loader, device)
    print(
        f"Тест: MSE={test_metrics['mse']:.4f}, MAE={test_metrics['mae']:.4f}, "
        f"R2={test_metrics['r2']:.4f} (остановка на эпохе {len(history['train_loss'])})"
    )

    plot_curves(
        {"train MSE": history["train_loss"], "val MSE": history["val_loss"]},
        title="Insurance: кривые обучения линейной регрессии",
        ylabel="MSE, станд.",
        save_path=PLOTS_DIR / "insurance_training.png",
    )
    with torch.no_grad():
        y_pred = model(test_ds.X.to(device)).cpu().numpy().ravel()
    _plot_regression_fit(
        test_ds.y.numpy().ravel(), y_pred,
        PLOTS_DIR / "insurance_fit.png",
        "Insurance: предсказание против истины на тесте",
    )
    torch.save(model.state_dict(), MODELS_DIR / "insurance_linreg.pth")
    print("Графики: insurance_training.png, insurance_fit.png; модель: insurance_linreg.pth\n")
    return test_metrics


def run_classification_experiment() -> Dict[str, float]:
    "Логистическая регрессия на titanic.csv"
    print("\n2.2 Классификация: выживание на Титанике (titanic.csv)")

    train_ds, test_ds = load_csv_splits(
        DATA_DIR / "titanic.csv",
        target_column="Survived",
        task="classification",
        numeric_cols=["Age", "SibSp", "Parch", "Fare"],
        binary_cols=["Sex"],
        categorical_cols=["Pclass", "Embarked"],
        drop_cols=["PassengerId", "Name", "Ticket", "Cabin"],
        seed=42,
    )
    num_classes = train_ds.num_classes
    logger.info(
        "titanic: train=%d, test=%d, признаков=%d, классов=%d",
        len(train_ds), len(test_ds), train_ds.n_features, num_classes,
    )
    print(f"Признаки: {train_ds.feature_names}")

    train_loader, val_loader = make_loaders(train_ds.X, train_ds.y, batch_size=32, val_ratio=0.2)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    set_seed(42)
    model = LogisticRegressionModel(in_features=train_ds.n_features, num_classes=num_classes)
    train_logistic_regression(
        model, train_loader, val_loader,
        num_classes=num_classes, epochs=300, lr=0.02, optimizer_name="Adam", patience=40, verbose=False,
    )

    device = get_device()
    metrics = evaluate_classification(model, test_loader, num_classes, device)
    print(
        f"Тест: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}"
    )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=["Погиб", "Выжил"],
        save_path=PLOTS_DIR / "titanic_confusion_matrix.png",
        title="Titanic: confusion matrix на тесте",
    )
    torch.save(model.state_dict(), MODELS_DIR / "titanic_logreg.pth")
    print("График: titanic_confusion_matrix.png; модель: titanic_logreg.pth\n")
    return {k: v for k, v in metrics.items() if isinstance(v, float)}


def main() -> None:
    set_seed(42)
    logger.info("Устройство: %s", get_device())
    run_regression_experiment()
    run_classification_experiment()
    print("Задание 2 выполнено.")


if __name__ == "__main__":
    main()
