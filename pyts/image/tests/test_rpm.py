"""Testing for Relative Position Matrix."""

# Author: Lucky J. Yang <jianqiy4@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.image import RelativePositionMatrix

@pytest.mark.parametrize(
    'x, k, arr_desired',
    [
        ([1, 2, 3, 4], 2,
         [[0.0, -255.0],
          [255.0, 0.0]]),

        ([5, 5, 5, 5, 5], 2,
         [[0.0, 0.0],
          [0.0, 0.0]])
    ]
)
def test_relative_position_matrix_expected(x, k, arr_desired):
    """Test that RPM gives expected matrices for simple cases."""
    arr_actual = RelativePositionMatrix(x, k)
    arr_desired = np.asarray(arr_desired, dtype=float)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-8, rtol=0.)

def test_invalid_inputs():
    """Test that invalid inputs raise errors."""
    with pytest.raises(Exception):
        RelativePositionMatrix("not an array", 2)
    with pytest.raises(Exception):
        RelativePositionMatrix([1, 2, 3], 0)
