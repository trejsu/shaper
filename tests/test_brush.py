import numpy as np

from shapes.shape import Brush


def test_params_intervals():
    w = 100
    h = 200
    expected = np.array([w, h, w, h, w, h, min(w, h) // 4])
    actual = Brush.params_intervals()(w=w, h=h)
    assert np.array_equal(expected, actual)
