"""Vectorized compression algorithms."""
import bz2
import gzip
import lzma

import numpy as np
from collections.abc import Callable


def vectorize_compressor(func: Callable[[bytes], bytes]) -> Callable[[bytes], bytes]:
    return np.vectorize(func, otypes=[bytes])


algorithms = {
    "gzip": vectorize_compressor(gzip.compress),
    "bzip2": vectorize_compressor(bz2.compress),
    "lzma": vectorize_compressor(lzma.compress),
}
