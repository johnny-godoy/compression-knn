"""Classify a text using KNN on a compression metric."""
from __future__ import annotations

import contextlib
import functools
import warnings

import numpy as np
import numpy.typing as npt
import scipy.stats
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Integral
from sklearn.utils._param_validation import Interval
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_random_state

from compression_knn._compression_algos import algorithms
from compression_knn.utils import compression_length
from compression_knn.utils import mode


with contextlib.suppress(ImportError):  # not in Python < 3.11
    from typing import Self


valid_algorithms = set(algorithms.keys())


class CompressionKNNClassifier(BaseEstimator, ClassifierMixin):
    r"""A KNN test classifier that uses gzip compression.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use by default for nearest neighbors queries.
    compressor : str, options={"gzip", "bzip2", "lzma"}, default="gzip"
        The compression algorithm to use.
    random_state : int, optional
        The seed of the pseudo random number generator used when breaking ties.

    References
    ----------
    [“Low-Resource” Text Classification: A Parameter-Free
     Classification Method with Compressors
     ](https://aclanthology.org/2023.findings-acl.426) (Jiang et al., Findings 2023)

    """
    _parameter_constraints = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "compressor": [StrOptions(valid_algorithms)],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        compressor: str = "gzip",
        random_state: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.compressor = compressor

    def _check_mode(self, n_classes: int):
        if self.n_neighbors % n_classes == 0:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is divisible"
                f" by n_classes ({n_classes})."
                " This may cause ties to be broken randomly, and is not advised."
            )
            return functools.partial(mode, rng=self._rng)
        return lambda x: scipy.stats.mode(x, axis=0)[0]

    def _check_params(self, n_samples: int) -> None:
        """Check the parameters passed to __init__."""
        self._validate_params()
        if self.n_neighbors > n_samples:
            raise ValueError(
                "Expected n_neighbors <= n_samples,"
                f" got {self.n_neighbors} > {n_samples}."
            )
        self._rng = check_random_state(self.random_state)

    def fit(self, X: npt.ArrayLike[str], y: npt.ArrayLike[str]) -> Self:
        """Fit the k-nearest neighbors classifier under the gzip compression metric.

        Parameters
        ----------
        X : npt.ArrayLike[str]
            The training data.
        y : npt.ArrayLike[str]
            The training labels.

        Returns
        -------
        Self
            The fitted estimator.

        """
        self.X_, self.y_ = self._validate_data(
            X, y, accept_sparse=False, ensure_2d=False, dtype="str"
        )
        self.X_ = self.X_.reshape((-1, 1))
        self._check_params(len(self.X_))
        self._encoder = LabelEncoder().fit(self.y_)
        self.y_ = self._encoder.transform(self.y_)
        self._mode = self._check_mode(len(self._encoder.classes_))
        self.train_lengths_ = compression_length(self.X_)
        return self  # type: ignore

    def _distance_matrix(self, X: npt.ArrayLike[str]) -> np.ndarray:
        """Return the distance matrix between X and the training data."""
        to_arr = check_array(X, dtype="str", ensure_2d=False)
        combination_matrix = np.char.add(to_arr, self.X_)
        combined_lengths = compression_length(combination_matrix)
        test_lengths = compression_length(to_arr)
        max_lengths = np.maximum(self.train_lengths_, test_lengths)
        min_lengths = np.minimum(self.train_lengths_, test_lengths)
        distances = (combined_lengths - min_lengths) / max_lengths
        return distances

    def predict(self, X: npt.ArrayLike[str]) -> np.ndarray[str]:
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : npt.ArrayLike[str]
            The data to predict.

        Returns
        -------
        np.ndarray[str]
            The predicted class labels.

        """
        distances = self._distance_matrix(X)
        indices = np.argpartition(distances, self.n_neighbors - 1, axis=0)[
            : self.n_neighbors
        ]
        most_common_indexes = self._mode(self.y_[indices])
        return self._encoder.inverse_transform(self.y_[most_common_indexes])


class CompressionKNNClassifierCV(BaseEstimator, ClassifierMixin):
    r"""See `CompressionKNNClassifier`.
    Implement fast hyperparameter tuning for n_neighbors,
      by reusing the distance matrix between the training data and the test data.

    Parameters
    ----------
    n_neighbors : list[int], default=[2, 3, 5, 10]
        Number of neighbors to use by default for nearest neighbors queries.
    compression_function: str, options={"gzip", "bzip2", "lzma"}
        The compression function to use.
    random_state : int, optional
        The seed of the pseudo random number generator used when breaking ties.
    cv: int or cross-validation generator, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold stratified cross validation,
        - int, to specify the number of folds in a (Stratified)KFold,
        - cross-validation generator.
    search_strategy: str, options={"partition", "sort"}
        The search strategy to use when finding the nearest neighbors.
        'partition' uses np.argpartition to find the n_neighbors smallest distances.
        It should be faster than 'sort' when the list of n_neighbors is small.
        'sort' uses np.argsort to find the n_neighbors smallest distances.
        It should be faster than 'partition' when the list of n_neighbors is large.

    """
    # TODO: Implement this
    ...
