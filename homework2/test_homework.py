"""Unit-тесты

Запуск:
    python -m unittest test_homework -v
    или
    python test_homework.py

Проверяются:
    метрики классификации из задание 1.2;
    логика ранней остановки и штрафов регуляризации из задание 1.1;
    предобработка в CSVDataset из задание 2.1;
    функции feature engineering из задание 3.2.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    f1_score as sk_f1,
    precision_score as sk_precision,
    recall_score as sk_recall,
    roc_auc_score as sk_roc_auc,
)

from homework_datasets import CSVDataset, load_csv_splits
from homework_experiments import (
    add_interaction_features,
    add_polynomial_features,
    add_statistical_features,
    build_engineered_features,
)
from homework_model_modification import (
    EarlyStopping,
    LinearRegressionModel,
    confusion_matrix,
    l1_penalty,
    l2_penalty,
    precision_recall_f1,
    roc_auc_score,
)
from utils import DATA_DIR


class TestClassificationMetrics(unittest.TestCase):
    "Метрики, реализованные с нуля, должны совпадать с scikit-learn"

    def _check_for_k(self, num_classes: int) -> None:
        rng = np.random.default_rng(num_classes)
        y_true = rng.integers(0, num_classes, size=400)
        scores = rng.random((400, num_classes))
        scores /= scores.sum(axis=1, keepdims=True)
        y_pred = scores.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred, num_classes)
        np.testing.assert_array_equal(cm, sk_confusion_matrix(y_true, y_pred))

        prf = precision_recall_f1(cm)
        self.assertAlmostEqual(prf["precision"], sk_precision(y_true, y_pred, average="macro", zero_division=0), places=6)
        self.assertAlmostEqual(prf["recall"], sk_recall(y_true, y_pred, average="macro", zero_division=0), places=6)
        self.assertAlmostEqual(prf["f1"], sk_f1(y_true, y_pred, average="macro", zero_division=0), places=6)

        auc = roc_auc_score(y_true, scores, num_classes)
        sk = sk_roc_auc(y_true, scores[:, 1] if num_classes == 2 else scores,
                        multi_class="ovr", average="macro")
        self.assertAlmostEqual(auc, sk, places=6)

    def test_binary_metrics(self):
        self._check_for_k(2)

    def test_multiclass_metrics(self):
        self._check_for_k(3)
        self._check_for_k(5)

    def test_perfect_prediction(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        cm = confusion_matrix(y, y, 3)
        prf = precision_recall_f1(cm)
        self.assertEqual(prf["precision"], 1.0)
        self.assertEqual(prf["recall"], 1.0)
        self.assertEqual(prf["f1"], 1.0)


class TestEarlyStopping(unittest.TestCase):
    "Проверка логики ранней остановки"

    def test_stops_after_patience(self):
        model = LinearRegressionModel(2)
        stopper = EarlyStopping(patience=3, min_delta=0.0)
        for epoch, loss in enumerate([1.0, 0.9, 0.8], start=1):
            self.assertFalse(stopper.step(loss, model, epoch))
        outcomes = [stopper.step(0.85, model, epoch) for epoch in range(4, 7)]
        self.assertEqual(outcomes, [False, False, True])
        self.assertEqual(stopper.best_epoch, 3)
        self.assertAlmostEqual(stopper.best_loss, 0.8)

    def test_restores_best_weights(self):
        torch.manual_seed(0)
        model = LinearRegressionModel(3)
        stopper = EarlyStopping(patience=2)
        stopper.step(1.0, model, 1)  
        best_weight = model.linear.weight.detach().clone()
        with torch.no_grad():
            model.linear.weight += 5.0 
        stopper.step(2.0, model, 2)
        stopper.step(2.0, model, 3)
        stopper.restore(model)
        self.assertTrue(torch.allclose(model.linear.weight, best_weight))


class TestRegularization(unittest.TestCase):
    "Штрафы L1/L2 считаются по весам"

    def test_penalty_values(self):
        model = LinearRegressionModel(3)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[1.0, -2.0, 3.0]]))
            model.linear.bias.copy_(torch.tensor([100.0])) 
        self.assertAlmostEqual(l1_penalty(model).item(), 6.0, places=5)   
        self.assertAlmostEqual(l2_penalty(model).item(), 14.0, places=5) 


class TestCSVDataset(unittest.TestCase):
    "Предобработка кастомного датасета"

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0, 5.0],
                "bin": ["yes", "no", "yes", "no", "yes"],
                "cat": ["a", "b", "c", "a", "b"],
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

    def test_numeric_normalization(self):
        ds = CSVDataset(
            self.df, "target", task="regression",
            numeric_cols=["num"], binary_cols=["bin"], categorical_cols=["cat"],
        )
        col = ds.X[:, 0].numpy()
        self.assertAlmostEqual(float(col.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(col.std()), 1.0, places=5)

    def test_feature_layout_and_no_nans(self):
        ds = CSVDataset(
            self.df, "target", task="regression",
            numeric_cols=["num"], binary_cols=["bin"], categorical_cols=["cat"],
        )
        self.assertEqual(ds.n_features, 5)
        self.assertEqual(len(ds.feature_names), 5)
        self.assertFalse(torch.isnan(ds.X).any())

    def test_binary_and_onehot_encoding(self):
        ds = CSVDataset(
            self.df, "target", task="regression",
            numeric_cols=["num"], binary_cols=["bin"], categorical_cols=["cat"],
        )
        binary_col = ds.X[:, 1]
        self.assertTrue(set(binary_col.tolist()) <= {0.0, 1.0})
        onehot = ds.X[:, 2:5]
        self.assertTrue(torch.all(onehot.sum(dim=1) == 1.0))

    def test_missing_values_imputed(self):
        df = self.df.copy()
        df.loc[0, "num"] = np.nan
        df.loc[1, "cat"] = np.nan
        ds = CSVDataset(
            df, "target", task="regression",
            numeric_cols=["num"], binary_cols=["bin"], categorical_cols=["cat"],
        )
        self.assertFalse(torch.isnan(ds.X).any())

    def test_classification_target(self):
        ds = CSVDataset(
            self.df.assign(target=[0, 1, 0, 1, 0]), "target", task="classification",
            numeric_cols=["num"], binary_cols=["bin"], categorical_cols=["cat"],
        )
        self.assertEqual(ds.y.dtype, torch.long)
        self.assertEqual(ds.num_classes, 2)

    def test_auto_type_inference(self):
        ds = CSVDataset(self.df, "target", task="regression")
        self.assertIn("num", ds.state.numeric_cols)
        self.assertIn("bin", ds.state.binary_cols)
        self.assertIn("cat", ds.state.categorical_cols)

    def test_train_test_consistency(self):
        path = DATA_DIR / "insurance.csv"
        if not path.exists():
            self.skipTest("insurance.csv отсутствует")
        train_ds, test_ds = load_csv_splits(
            path, "charges", "regression",
            numeric_cols=["age", "bmi", "children"],
            binary_cols=["sex", "smoker"], categorical_cols=["region"],
        )
        self.assertEqual(train_ds.n_features, test_ds.n_features)
        self.assertIs(train_ds.state, test_ds.state)  
        self.assertFalse(torch.isnan(test_ds.X).any())


class TestFeatureEngineering(unittest.TestCase):
    "Формы инженерных признаков"

    def setUp(self):
        self.X = torch.randn(20, 5)
        self.numeric_idx = [0, 1, 2]

    def test_polynomial_shape(self):
        poly = add_polynomial_features(self.X, self.numeric_idx, degree=2)
        self.assertEqual(poly.shape, (20, 3))  
        self.assertTrue(torch.allclose(poly[:, 0], self.X[:, 0] ** 2))

    def test_interaction_shape(self):
        inter = add_interaction_features(self.X, self.numeric_idx)
        self.assertEqual(inter.shape, (20, 3)) 
        self.assertTrue(torch.allclose(inter[:, 0], self.X[:, 0] * self.X[:, 1]))

    def test_statistical_shape(self):
        stats = add_statistical_features(self.X, self.numeric_idx)
        self.assertEqual(stats.shape, (20, 4))  
        self.assertTrue(torch.allclose(stats[:, 0], self.X[:, self.numeric_idx].mean(dim=1)))

    def test_build_engineered_grows_feature_count(self):
        eng = build_engineered_features(self.X, self.numeric_idx, interaction_idx=[0, 1, 2, 3, 4])
        self.assertEqual(eng.shape, (20, 22))


if __name__ == "__main__":
    unittest.main(verbosity=2)
