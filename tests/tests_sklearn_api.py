"""Tests whether TextKNNClassifier follows the sklearn API."""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from compression_knn.knn import CompressionKNNClassifier


def generate_data():
    """Make a simple text classification dataset."""
    X = [
        "red, round, sweet",
        "orange, round, tangy",
        "red, oblong, sweet",
        "orange, oblong, tangy",
        "green, round, sour",
        "orange, round, sweet",
        "green, oblong, sour",
        "red, round, sour",
        "yellow, round, sweet",
        "green, oval, tangy",
        "yellow, oval, sweet",
        "orange, round, sour",
        "red, oblong, tangy",
        "yellow, round, tangy",
        "green, round, sweet",
    ]

    y = [
        "Apple",
        "Orange",
        "Apple",
        "Orange",
        "Apple",
        "Orange",
        "Apple",
        "Apple",
        "Banana",
        "Kiwi",
        "Banana",
        "Orange",
        "Apple",
        "Lemon",
        "Apple",
    ]
    return X, y


class TestSklearnAPICompressionKNNClassifier(unittest.TestCase):
    def setUp(self):
        self.X, self.y = generate_data()
        self.classifier = CompressionKNNClassifier(n_neighbors=3)

    def test_is_sklearn_classifier(self):
        self.assertEqual(self.classifier._estimator_type, "classifier")

    def test_fit(self):
        self.classifier.fit(self.X, self.y)

    def test_predict(self):
        self.classifier.fit(self.X, self.y)
        y_pred = self.classifier.predict(self.X)
        self.assertEqual(len(y_pred), len(self.y))

    def test_clone(self):
        self.classifier.fit(self.X, self.y)
        cloned_regressor = clone(self.classifier)
        self.assertIsNot(self.classifier, cloned_regressor)
        self.assertEqual(self.classifier.get_params(), cloned_regressor.get_params())

    def test_pipeline(self):
        pipeline = Pipeline(
            [
                ("classifier", self.classifier),
            ]
        )
        pipeline.fit(self.X, self.y)
        pipeline.predict(self.X)

    def test_grid_search(self):
        param_grid = {
            "n_neighbors": [2, 3, 5],
        }
        grid_search = GridSearchCV(
            self.classifier,
            param_grid,
            cv=3,
        )
        grid_search.fit(np.array(self.X).reshape(-1, 1), np.array(self.y))


if __name__ == "__main__":
    unittest.main()
