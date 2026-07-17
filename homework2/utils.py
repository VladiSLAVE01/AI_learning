# Общие утилиты


from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"

for _directory in (DATA_DIR, MODELS_DIR, PLOTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str) -> logging.Logger:
    """Создаёт или возвращает готовый логгер с единым форматом вывода

    Повторные вызовы с тем же name не добавляют дублирующих обработчиков
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def plot_curves(
    history: Mapping[str, Sequence[float]],
    title: str,
    ylabel: str,
    save_path: Path | str,
    xlabel: str = "Эпоха",
    logy: bool = False,
) -> Path:
    "Рисует несколько кривых на одном графике и сохраняет его в файл"
    save_path = Path(save_path)
    plt.figure(figsize=(8, 5))
    for label, values in history.items():
        plt.plot(range(1, len(values) + 1), values, label=label)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


def plot_bars(
    values: Mapping[str, float],
    title: str,
    ylabel: str,
    save_path: Path | str,
    xlabel: str = "",
) -> Path:
    "Строит столбчатую диаграмму по словарю {подпись: значение}"
    save_path = Path(save_path)
    plt.figure(figsize=(8, 5))
    labels = list(values.keys())
    heights = list(values.values())
    bars = plt.bar(labels, heights, color="#4C72B0")
    for bar, height in zip(bars, heights):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.4g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path
