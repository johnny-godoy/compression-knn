"""Preprocessor utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array


class VectorToTextTransformer(TransformerMixin, BaseEstimator):
    """Convert each feature vector row into a single text sample.

    This transformer is useful when a downstream estimator expects textual
    inputs, but the original data is represented as vectors of numbers or mixed
    values.

    Parameters
    ----------
    separator : str, default=" "
        Token inserted between values from the same row.

    """

    def __init__(self, separator: str = " "):
        self.separator = separator

    def _validate_input(self, X: npt.ArrayLike) -> np.ndarray:
        arr = check_array(X, ensure_2d=False, dtype=None)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("X must be a 1D or 2D array-like input.")
        return arr

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        """Validate input and remember feature dimensionality."""
        arr = self._validate_input(X)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> np.ndarray:
        """Return one text sample per input row."""
        arr = self._validate_input(X)
        if hasattr(self, "n_features_in_") and arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "Number of features in transform differs from fit: "
                f"{arr.shape[1]} != {self.n_features_in_}."
            )
        as_object = np.asarray(arr, dtype=object)
        return np.array(
            [self.separator.join(map(str, row)) for row in as_object],
            dtype=str,
        )
