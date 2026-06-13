"""Unit tests for preprocessing utilities."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from sklearn.pipeline import Pipeline

from compression_knn.knn import CompressionKNNClassifier
from compression_knn.preprocessor import VectorToTextTransformer


class TestVectorToTextTransformer(unittest.TestCase):
    def test_transform_1d_input(self):
        transformer = VectorToTextTransformer()
        X = np.array([1, 2, 3])

        transformed = transformer.fit_transform(X)

        assert_array_equal(transformed, np.array(["1", "2", "3"]))

    def test_transform_2d_input_with_custom_separator(self):
        transformer = VectorToTextTransformer(separator="|")
        X = np.array([[1, 2.5, True], ["red", None, "x"]], dtype=object)

        transformed = transformer.fit_transform(X)

        assert_array_equal(
            transformed,
            np.array(["1|2.5|True", "red|None|x"]),
        )

    def test_transform_raises_on_feature_mismatch(self):
        transformer = VectorToTextTransformer().fit(np.array([[1, 2], [3, 4]]))

        with self.assertRaises(ValueError):
            transformer.transform(np.array([[1, 2, 3]]))

    def test_pipeline_with_classifier(self):
        X_train = np.array([[1, 1], [1, 2], [9, 9], [8, 9]])
        y_train = np.array(["low", "low", "high", "high"])
        X_test = np.array([[1, 0], [9, 8]])

        pipeline = Pipeline(
            [
                ("vector_to_text", VectorToTextTransformer(separator=",")),
                ("classifier", CompressionKNNClassifier(n_neighbors=1)),
            ]
        )
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        self.assertEqual(len(predictions), 2)
        self.assertTrue(set(predictions).issubset({"low", "high"}))


if __name__ == "__main__":
    unittest.main()