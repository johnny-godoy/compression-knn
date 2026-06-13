"""Test performance on a simple dataset."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from compression_knn._compression_algos import algorithms
from compression_knn.knn import CompressionKNNClassifier, CompressionKNNClassifierCV

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

    def test_predict_multiple_samples_multiclass(self):
        X_train = [
            "alpha apple crisp",
            "beta orange tangy",
            "gamma banana mellow",
        ]
        y_train = ["Apple", "Orange", "Banana"]
        X_test = [
            "alpha apple crisp",
            "beta orange tangy",
            "gamma banana mellow",
        ]

        model = CompressionKNNClassifier(n_neighbors=1)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert_array_equal(predictions, np.array(y_train))

    def test_cv_classifier_predicts_string_labels(self):
        X_train = [
            "alpha apple crisp",
            "alpha apple tart",
            "beta orange tangy",
            "beta orange peel",
            "gamma banana mellow",
            "gamma banana split",
        ]
        y_train = ["Apple", "Apple", "Orange", "Orange", "Banana", "Banana"]

        model = CompressionKNNClassifierCV(n_neighbors=[1], cv=2, random_state=0)
        model.fit(X_train, y_train)

        predictions = model.predict([
            "alpha apple crisp",
            "beta orange tangy",
            "gamma banana mellow",
        ])

        assert_array_equal(predictions, np.array(["Apple", "Orange", "Banana"]))

    def test_cv_partition_matches_sort_for_k1(self):
        X_train = [
            "aaaa apple",
            "aaa apple",
            "bbbb orange",
            "bbb orange",
            "cccc banana",
            "ccc banana",
        ]
        y_train = ["Apple", "Apple", "Orange", "Orange", "Banana", "Banana"]

        sort_model = CompressionKNNClassifierCV(
            n_neighbors=[1],
            cv=2,
            search_strategy="sort",
            random_state=0,
        )
        partition_model = CompressionKNNClassifierCV(
            n_neighbors=[1],
            cv=2,
            search_strategy="partition",
            random_state=0,
        )

        sort_model.fit(X_train, y_train)
        partition_model.fit(X_train, y_train)

        assert_array_equal(partition_model.cv_result_, sort_model.cv_result_)
        self.assertEqual(partition_model.n_neighbors_, sort_model.n_neighbors_)
        assert_array_equal(
            partition_model.predict(X_train),
            sort_model.predict(X_train),
        )


if __name__ == "__main__":
    unittest.main()
