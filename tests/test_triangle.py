import numpy as np

from shapes.shape import Triangle


def test_from_normalized_params():
    normalized = [1, 0.2866666666666667, 0.85, 0.32666666666666666, 0.65, 0.013333333333333334, 1, 2, 3, 0.5]
    w = 100
    h = 150
    expected_points = np.array([[100, 43], [85, 49], [65, 2]])
    expected_color = np.array([1, 2, 3])
    r = Triangle.from_normalized_params(w, h, *normalized)
    assert np.array_equal(r.points, expected_points)
    assert r.alpha == 0.5
    assert np.array_equal(r.color, expected_color)


def test_normalized_params():
    t = Triangle.from_params(*[100, 43, 85, 49, 65, 2, 1, 2, 3, 0.5])
    w = 100
    h = 150
    expected = np.array(
        [1, 0.2866666666666667, 0.85, 0.32666666666666666, 0.65, 0.013333333333333334, 1, 2, 3, 0.5])
    normalized = t.normalized_params(w, h)
    assert np.array_equal(normalized, expected)
