"""Vectorized compression algorithms."""
import bz2
import gzip
import lzma

import numpy as np


def vectorize_compressor(func: callable) -> callable:
    return np.vectorize(func, otypes=[bytes])


algorithms = {
    "gzip": vectorize_compressor(gzip.compress),
    "bzip2": vectorize_compressor(bz2.compress),
    "lzma": vectorize_compressor(lzma.compress),
}
