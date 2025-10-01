"""Testing for Relative Angle Matrix (RAM)."""

# Author: Lucky J. Yang <jianqiy4@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from pyts.image import RelativeAngleMatrix


@pytest.mark.parametrize(
    "V, d, expected_shape",
    [
        ([1, 2, 3, 4], 2, (2, 2)),
        ([1, 2, 3, 4, 5, 6], 3, (2, 2)),
        ([0, 0, 0, 0], 2, (2, 2))
    ]
)
def test_output_shape(V, d, expected_shape):
    """Test that RAM returns a square matrix of correct shape."""
    A = RelativeAngleMatrix(V, d)
    assert isinstance(A, np.ndarray)
    assert A.shape == expected_shape


def test_range_and_scaling():
    """Test that output is always scaled to [0, 255]."""
    V = np.linspace(0, 10, 20)
    A = RelativeAngleMatrix(V, d=4)
    assert np.all(A >= 0)
    assert np.all(A <= 255)
    assert np.isclose(A.min(), 0.0, atol=1e-8)
    assert np.isclose(A.max(), 255.0, atol=1e-8)


@pytest.mark.parametrize("length,d", [(10, 2), (15, 3), (50, 5)])
def test_symmetry_and_diagonal(length, d):
    """Test that RAM matrix has zero diagonal and is anti-symmetric before scaling."""
    V = np.random.RandomState(0).randn(length)
    A = RelativeAngleMatrix(V, d)
    A01 = (A / 255.0)
    diag = np.diag(A01)
    assert np.allclose(diag, 0.5, atol=1e-2)


def test_invalid_inputs():
    """Test invalid inputs raise exceptions."""
    with pytest.raises(Exception):
        RelativeAngleMatrix("not array", 2)
    with pytest.raises(Exception):
        RelativeAngleMatrix([1, 2, 3], 0)
