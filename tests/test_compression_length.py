"""Unit tests for the compression_length function."""
import bz2
import gzip
import lzma
import unittest

import numpy as np

from compression_knn.utils import compression_length
from tests.test_assert_less_equal import assert_array_less_equal


class TestGzipCompressionLength(unittest.TestCase):
    @staticmethod
    def assert_array_less_equal(x, y, err_msg="", verbose=True):
        return assert_array_less_equal(x, y, err_msg=err_msg, verbose=verbose)

    def test_empty_texts_gzip(self):
        texts = np.array([], dtype=str)
        expected_lengths = np.array([], dtype=int)
        compressed_lengths = compression_length(texts)
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_empty_texts_bzip2(self):
        texts = np.array([], dtype=str)
        expected_lengths = np.array([], dtype=int)
        compressed_lengths = compression_length(texts, algorithm="bzip2")
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_empty_texts_lzma(self):
        texts = np.array([], dtype=str)
        expected_lengths = np.array([], dtype=int)
        compressed_lengths = compression_length(texts, algorithm="lzma")
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_single_text_gzip(self):
        texts = np.array(["This is a single text."], dtype=str)
        expected_lengths = np.array(
            [len(gzip.compress(texts[0].encode("utf-8")))], dtype=int
        )
        compressed_lengths = compression_length(texts)
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_single_text_bzip2(self):
        texts = np.array(["This is a single text."], dtype=str)
        expected_lengths = np.array(
            [len(bz2.compress(texts[0].encode("utf-8")))], dtype=int
        )
        compressed_lengths = compression_length(texts, algorithm="bzip2")
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_single_text_lzma(self):
        texts = np.array(["This is a single text."], dtype=str)
        expected_lengths = np.array(
            [len(lzma.compress(texts[0].encode("utf-8")))], dtype=int
        )
        compressed_lengths = compression_length(texts, algorithm="lzma")
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_multiple_texts_gzip(self):
        texts = np.array(["Text 1", "Text 2", "Text 3"], dtype=str)
        expected_lengths = np.array(
            [len(gzip.compress(t.encode("utf-8"))) for t in texts], dtype=int
        )
        compressed_lengths = compression_length(texts)
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_multiple_texts_bzip2(self):
        texts = np.array(["Text 1", "Text 2", "Text 3"], dtype=str)
        expected_lengths = np.array(
            [len(bz2.compress(t.encode("utf-8"))) for t in texts], dtype=int
        )
        compressed_lengths = compression_length(texts, algorithm="bzip2")
        self.assert_array_less_equal(compressed_lengths, expected_lengths)

    def test_multiple_texts_lzma(self):
        texts = np.array(["Text 1", "Text 2", "Text 3"], dtype=str)
        expected_lengths = np.array(
            [len(lzma.compress(t.encode("utf-8"))) for t in texts], dtype=int
        )
        compressed_lengths = compression_length(texts, algorithm="lzma")
        self.assert_array_less_equal(compressed_lengths, expected_lengths)


if __name__ == "__main__":
    unittest.main()
