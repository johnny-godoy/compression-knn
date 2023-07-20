"""Test performance on a simple dataset."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from compression_knn._compression import algorithms
from compression_knn.knn import CompressionKNNClassifier


compressors = list(algorithms.keys())


class TestCompressionKNNClassifier(unittest.TestCase):
    def test_fit_and_predict(self):
        X_train = [
            "red, round, sweet",
            "orange, round, tangy",
            "red, oblong, sweet",
            "orange, oblong, tangy",
            "green, round, sour",
        ]
        y_train = ["Apple", "Orange", "Apple", "Orange", "Apple"]
        X_test = ["yellow, round, sweet", "green, round, sweet"]
        expected_predictions = ["Apple", "Apple"]

        for algorithm in compressors:
            model = CompressionKNNClassifier(n_neighbors=3, compressor=algorithm)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            assert_array_equal(predictions, expected_predictions)

    def test_predict_single_instance(self):
        X_train = [
            "red, round, sweet",
            "orange, round, tangy",
            "red, oblong, sweet",
            "orange, oblong, tangy",
            "green, round, sour",
        ]
        y_train = ["Apple", "Orange", "Apple", "Orange", "Apple"]

        model = CompressionKNNClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        single_instance = ["yellow, round, sweet"]
        prediction = model.predict(single_instance)

        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(len(prediction), 1)


if __name__ == "__main__":
    unittest.main()
