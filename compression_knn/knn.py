"""Classify a text using KNN on a compression metric."""
from __future__ import annotations

import abc
import contextlib
import functools
import warnings

import numpy as np
import numpy.typing as npt
import scipy.stats
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import check_scoring
from sklearn.metrics import get_scorer_names
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
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


class BaseCompressionKNN(BaseEstimator, ClassifierMixin, abc.ABC):
    _parameter_constraints = {
        "compressor": [StrOptions(valid_algorithms)],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        n_neighbors,
        compressor: str = "gzip",
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.compressor = compressor
        self.random_state = random_state

    @abc.abstractmethod
    def _check_neighbors(self, n_neighbors, n_samples: int):
        ...

    @abc.abstractmethod
    def _check_mode(self, n_neighbors: int, n_classes: int) -> callable:
        ...

    def _check_params(self):
        self._validate_params()
        self._rng = check_random_state(self.random_state)

    def fit(self, X: npt.ArrayLike[str], y: npt.ArrayLike[str]) -> Self:
        self._check_params()
        self.X_, self.y_ = self._validate_data(
            X, y, accept_sparse=False, ensure_2d=False, dtype="str"
        )
        self.X_ = self.X_.reshape((-1, 1))
        self._encoder = LabelEncoder().fit(self.y_)
        self.y_ = self._encoder.transform(self.y_)
        self._n_neighbors = self._check_neighbors(self.n_neighbors, len(self.X_))
        self._mode = self._check_mode(self._n_neighbors, len(self._encoder.classes_))
        self.train_lengths_ = compression_length(self.X_, self.compressor)
        return self  # type: ignore

    def _distance_matrix(self, X: npt.ArrayLike[str]) -> np.ndarray:
        """Return the distance matrix between X and the training data."""
        to_arr = check_array(X, dtype="str", ensure_2d=False)
        combination_matrix = np.char.add(to_arr, self.X_)
        combined_lengths = compression_length(combination_matrix, self.compressor)
        test_lengths = compression_length(to_arr, self.compressor)
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
        indices = np.argpartition(distances, self._n_neighbors - 1, axis=0)[
            : self._n_neighbors
        ]
        most_common_indexes = self._mode(self.y_[indices])
        return self._encoder.inverse_transform(self.y_[most_common_indexes])


class CompressionKNNClassifier(BaseCompressionKNN):
    r"""A KNN test classifier that uses gzip compression.

    Parameters
    ----------
    n_neighbours : int, default=5
        Number of neighbors to use by default for nearest neighbors queries.
    compressor : str, options={"gzip", "bzip2", "lzma"}, default="gzip"
        The compression algorithm to use.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. It's used to break ties when
        multiple classes have the same number of votes. Pass an int for reproducible
        output across multiple function calls if `n_neighbors` is divisible by
        `n_classes`.

    Attributes
    ----------
    X_ : np.ndarray[str]
        The training data.
    y_ : np.ndarray[str]
        The training labels.
    train_lengths_ : np.ndarray[int]
        The lengths of the compressed training data.
    n_neighbors_ : int
        Copy of the parameter with the same name. Used at predict time.

    References
    ----------
    [“Low-Resource” Text Classification: A Parameter-Free
     Classification Method with Compressors
     ](https://aclanthology.org/2023.findings-acl.426) (Jiang et al., Findings 2023)

    """
    _parameter_constraints = {
        **BaseCompressionKNN._parameter_constraints,
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
    }

    def _check_neighbors(self, n_neighbors, n_samples: int):
        if n_neighbors > n_samples:
            raise ValueError(
                f"n_neighbors ({n_neighbors}) must be less than or equal to"
                f" the number of samples ({n_samples})."
            )
        return n_neighbors

    def _check_mode(self, n_neighbors, n_classes: int) -> callable:
        if n_neighbors % n_classes == 0:
            warnings.warn(
                f"n_neighbors ({n_neighbors}) is divisible by the number of classes"
                f" ({n_classes}). This means that ties will be broken randomly."
            )
            return functools.partial(mode, rng=self._rng)
        return lambda x: scipy.stats.mode(x, axis=0)[0]


class CompressionKNNClassifierCV(BaseCompressionKNN):
    r"""See `CompressionKNNClassifier`.
    Implement fast hyperparameter tuning for n_neighbors,
      by reusing the distance matrix between the training data and the test data.

    Parameters
    ----------
    n_neighbors : list[int], default=np.array([2, 3, 5, 10])
        Number of neighbors to use by default for nearest neighbors queries.
    compressor: str, options={"gzip", "bzip2", "lzma"}
        The compression function to use.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. It's used to break ties when
        multiple classes have the same number of votes. Pass an int for reproducible
        output across multiple function calls if `n_neighbors` is divisible by
        `n_classes`.
    cv: int, cross-validation generator or iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold stratified cross validation,
        - int, to specify the number of folds in a StratifiedKFold,
        - CV splitter
        - An iterable yielding (train, test) splits as arrays of indices.
    scoring: str, callable, or None, default=None
        A string (see sklearn's model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    search_strategy: str, options={"partition", "sort"}
        The search strategy to use when finding the nearest neighbors.
        'partition' uses np.argpartition to find the n_neighbors smallest distances.
        It should be faster than 'sort' when the list of n_neighbors is small.
        'sort' uses np.argsort to find the n_neighbors smallest distances.
        It should be faster than 'partition' when the list of n_neighbors is large.

    Attributes
    ----------
    X_ : np.ndarray[str]
        The training data.
    y_ : np.ndarray[str]
        The training labels.
    train_lengths_ : np.ndarray[int]
        The lengths of the compressed training data.
    cv_result_ : np.ndarray[float] of shape (len(n_neighbors), len(cv))
        All scores obtained by cross-validation.
    n_neighbors_ : int
        The best n_neighbors found by cross-validation.
    best_score_ : float
        The best score found by cross-validation.

    """
    _parameter_constraints: dict = {
        **BaseCompressionKNN._parameter_constraints,
        "n_neighbors": ["array-like"],
        "cv": ["cv_object"],
        "search_strategy": [
            StrOptions(
                {
                    "partition",
                    "sort",
                }
            )
        ],
        "scoring": [
            StrOptions(set(get_scorer_names())),
            callable,
            None,
        ],
    }

    def __init__(
        self,
        n_neighbors: np.ndarray[int] = np.array([2, 3, 5, 10]),
        compressor: str = "gzip",
        random_state: int | np.random.RandomState | None = None,
        cv: int | BaseCrossValidator | None = None,
        scoring: str | callable | None = None,
        search_strategy: str = "partition",
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            compressor=compressor,
            random_state=random_state,
        )
        self.cv = cv
        self.scoring = scoring
        self.search_strategy = search_strategy

    def _check_neighbors(self, n_neighbors, n_samples: int) -> np.ndarray[int]:
        arr = check_array(n_neighbors, ensure_2d=False, dtype=int)
        max_neighbors = np.max(arr)  # type: ignore
        if max_neighbors > n_samples:
            raise ValueError(
                f"The largest n_neighbors {max_neighbors} is larger"
                f" than the smallest training fold size {n_samples}."
            )
        min_neighbors = np.min(arr)  # type: ignore
        if min_neighbors < 1:
            raise ValueError(
                f"The smallest n_neighbors {min_neighbors} is less than 1."
            )
        return arr  # type: ignore

    def _check_mode(self, n_neighbors, n_classes: int) -> callable:
        remainders = np.remainder(n_neighbors, n_classes)
        if np.any(remainders == 0):
            warnings.warn(
                f"A value of n_neighbors is divisible by n_classes ({n_classes})."
                " This is not recommended as it can lead to ties."
            )
            return functools.partial(mode, rng=self._rng)
        return lambda x: scipy.stats.mode(x, axis=0)[0]  # type: ignore

    def _check_params(self):
        super()._check_params()
        self._scorer = check_scoring(self, self.scoring, allow_none=True)
        if self._scorer is None:
            self._scorer = accuracy_score

    def fit(self, X: npt.ArrayLike[str], y: npt.ArrayLike[str]) -> Self:
        super().fit(X, y)
        # Now we need to improve the checking
        self._cv = check_cv(self.cv, y, classifier=True)
        self._check_params()
        # Finish checking the neighbors
        n_samples = np.min([len(fold) for _, fold in self._cv.split(X, y)])
        self._n_neighbors = self._check_neighbors(self.n_neighbors, n_samples)
        self.cv_result_ = np.empty((len(self._n_neighbors), len(self._cv)))
        # Populating the cv_result_ array with the chosen search strategy
        if self.search_strategy == "sort":
            nearest = np.argsort(self._distance_matrix(self.X_), axis=1)
            for train, test in self._cv.split(X, y):
                y_train, y_test = y[train], y[test]
                for i, n_neighbors in enumerate(self._n_neighbors):
                    neighbors = nearest[test, :n_neighbors]
                    y_pred = self._mode(y_train[neighbors])
                    score = self._scorer(y_test, y_pred)
                    self.cv_result_[i, test] = score
        elif self.search_strategy == "partition":
            raise NotImplementedError(
                "The 'partition' search strategy is not implemented yet."
            )
        # Find the best n_neighbors given the cross-validation results
        mean_scores = np.mean(self.cv_result_, axis=1)
        best_index = np.argmax(mean_scores)
        self._n_neighbors = self._n_neighbors[best_index]
        self.n_neighbors_ = self._n_neighbors  # For user access
        self.best_score_ = mean_scores[best_index]
        return self
