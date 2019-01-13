import numpy as np

from shaper.shape.brush import Brush, EllipseBrush, merge_bounds_for_simple_path


def test_args_intervals():
    w = 100
    h = 200
    expected = np.array([w, h, w, h, w, h, min(w, h)])
    actual = Brush.args_intervals()(w=w, h=h)
    assert np.array_equal(expected, actual)


def test_args():
    brush = EllipseBrush.from_params(1, 2, 3, 4, 5, 6, 1, 10)
    expected = np.array([1, 2, 3, 4, 5, 6, 1])
    actual = brush.args()
    assert np.array_equal(expected, actual)


def test_merge_bounds():
    bounds = np.array([[1, 2, 3], [1, 3, 4], [2, 5, 3], [0, 2, 4], [1, 2, 5]])
    expected_merged = np.array([[1, 5, 3], [0, 3, 4], [1, 2, 5]])
    merged = merge_bounds_for_simple_path(bounds=bounds)
    assert np.array_equal(merged, expected_merged)