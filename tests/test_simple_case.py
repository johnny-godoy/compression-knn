"""Test performance on a simple dataset."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.model_selection import StratifiedKFold

from compression_knn._compression_algos import algorithms
from compression_knn.knn import CompressionKNNClassifier, CompressionKNNClassifierCV

compressors = list(algorithms.keys())


class TestCompressionKNNClassifier(unittest.TestCase):
    def test_fit_rejects_multicolumn_2d_input(self):
        X_train = np.array([["red", "round"], ["orange", "tangy"]])
        y_train = ["Apple", "Orange"]

        model = CompressionKNNClassifier(n_neighbors=1)
        with self.assertRaisesRegex(ValueError, "single column"):
            model.fit(X_train, y_train)

    def test_fit_with_single_column_2d_input_sets_n_neighbors_attribute(self):
        X_train = np.array(
            [
                "red, round, sweet",
                "orange, round, tangy",
                "red, oblong, sweet",
            ]
        ).reshape(-1, 1)
        y_train = ["Apple", "Orange", "Apple"]

        model = CompressionKNNClassifier(n_neighbors=2)
        model.fit(X_train, y_train)

        self.assertEqual(model.n_neighbors_, 2)

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

    def test_cv_scores_match_manual_classifier_cv_for_k1(self):
        X_train = np.array(["aba", "bbbb", "bab", "aab", "bbb", "aaaa"])
        y_train = np.array(["A", "B", "B", "A", "B", "A"])

        cv = StratifiedKFold(n_splits=3, shuffle=False)
        manual_scores = []
        for train, test in cv.split(X_train, y_train):
            model = CompressionKNNClassifier(n_neighbors=1, random_state=0)
            model.fit(X_train[train], y_train[train])
            y_pred = model.predict(X_train[test])
            manual_scores.append(np.mean(y_pred == y_train[test]))
        manual_scores = np.array(manual_scores)

        model = CompressionKNNClassifierCV(
            n_neighbors=[1],
            cv=3,
            search_strategy="sort",
            random_state=0,
        )
        model.fit(X_train, y_train)

        assert_array_equal(model.cv_result_[0], manual_scores)


if __name__ == "__main__":
    unittest.main()
