import numpy as np

from shaper.shape.brush import Brush, EllipseBrush, merge_bounds_for_simple_path


def test_params_intervals():
    w = 100
    h = 200
    expected = np.array([w, h, w, h, w, h, min(w, h) // 2])
    actual = Brush.params_intervals()(w=w, h=h)
    assert np.array_equal(expected, actual)


def test_params():
    brush = EllipseBrush.from_params(1, 2, 3, 4, 5, 6, 1, 10)
    expected = np.array([1, 2, 3, 4, 5, 6, 1])
    actual = brush.params()
    assert np.array_equal(expected, actual)


def test_merge_bounds():
    bounds = np.array([[1, 2, 3], [1, 3, 4], [2, 5, 3], [0, 2, 4], [1, 2, 5]])
    expected_merged = np.array([[1, 5, 3], [0, 3, 4], [1, 2, 5]])
    merged = merge_bounds_for_simple_path(bounds=bounds)
    assert np.array_equal(merged, expected_merged)


def test_from_normalized_params():
    normalized = [0.54, 0.246666667, 0.26, 0.06, 0.71, 0.346666667, 0.26, 0.5]
    w = 100
    h = 150
    expected_points = np.array([[54, 37], [26, 9], [71, 52]])
    expected_size = 13
    b = EllipseBrush.from_normalized_params(w, h, *normalized)
    assert np.array_equal(b.path.points, expected_points)
    assert b.size == expected_size


def test_normalized_params():
    b = EllipseBrush.from_params(*[54, 15, 26, 9, 71, 300, 13, 0.5])
    w = 100
    h = 150
    expected = np.array([0.54, 0.1, 0.26, 0.06, 0.71, 2, 0.26, 0.5])
    normalized = b.normalized_params(w, h)
    assert np.array_equal(normalized, expected)
