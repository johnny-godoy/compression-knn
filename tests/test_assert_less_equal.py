"""Implement and test a custom assert_array_less_equal function."""

import operator
import unittest

import numpy as np
from numpy.testing import assert_array_compare


def assert_array_less_equal(x, y, err_msg="", verbose=True):
    assert_array_compare(
        operator.__le__,
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        header="Arrays are not less than or equal ordered",
        equal_inf=False,
    )


class TestAssertArrayLessEqual(unittest.TestCase):
    @staticmethod
    def assert_array_less_equal(x, y, err_msg="", verbose=True):
        return assert_array_less_equal(x, y, err_msg=err_msg, verbose=verbose)

    def test_equal_arrays(self):
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        self.assert_array_less_equal(x, y)  # Should not raise any exception

    def test_less_ordered_arrays(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        self.assert_array_less_equal(x, y)  # Should not raise any exception

    def test_greater_ordered_arrays(self):
        x = np.array([4, 5, 6])
        y = np.array([1, 2, 3])
        with self.assertRaises(AssertionError):
            self.assert_array_less_equal(x, y)


if __name__ == "__main__":
    unittest.main()
