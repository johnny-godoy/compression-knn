"""Implement utility functions."""
import random
from typing import Optional

import numpy as np

from compression_knn._compression import algorithms


def compression_length(
    texts: np.ndarray[str],
    algorithm: str = "gzip",
) -> np.ndarray[int]:
    """Return the length of the compressed texts.

    Parameters
    ----------
    texts : np.ndarray[str]
        The texts to compress.
    algorithm : str, options={"gzip", "bzip2", "lzma"}, default="gzip"
        The compression algorithm to use.

    Returns
    -------
    np.ndarray[int]
        The length of the compressed texts.

    """
    encoded = np.char.encode(texts, encoding="utf-8")
    compressed = algorithms[algorithm](encoded)
    lengths = np.char.str_len(compressed)
    return lengths


def mode(
    arr: np.ndarray[int],
    rng: Optional[random.Random] = None,
) -> np.ndarray[int]:
    """Return the mode vector of a 2D array.

    Parameters
    ----------
    arr : np.ndarray[int, ndim=2]
        The array to find the mode vector of.
    rng : random.Random, optional
        The random number generator used for breaking ties. It must be given if there
        can be multiple modes, otherwise it can be omitted.

    Returns
    -------
    np.ndarray[int, ndim=1]
        The mode vector of the array.

    Raises
    ------
    ValueError
        If there are multiple modes and no RNG is given.

    """
    all_modes = []
    for col in arr.T:
        unique_elements, counts = np.unique(col, return_counts=True)
        max_count = np.max(counts)
        modes = unique_elements[counts == max_count]
        all_modes.append(modes)
    if all(len(md) == 1 for md in all_modes):
        return np.array(all_modes).flatten()
    if rng is None:
        raise ValueError("Multiple modes found but no RNG given.")
    return np.array([rng.choice(modes) for modes in all_modes])
