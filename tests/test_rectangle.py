import numpy as np

from shaper.shape.rectangle import Rectangle


def test_from_normalized_params():
    normalized = [1, 0.2866666666666667, 0.85, 0.32666666666666666, 0.65, 0.013333333333333334, 0.8,
                  -0.02, 0.5]
    w = 100
    h = 150
    expected_points = np.array([[100, 43], [85, 49], [65, 2], [80, -3]])
    r = Rectangle.from_normalized_params(w, h, *normalized)
    assert np.array_equal(r.points, expected_points)
    assert r.alpha == 0.5


def test_normalized_params():
    r = Rectangle.from_params(*[100, 43, 85, 49, 65, 2, 80, -3, 0.5])
    w = 100
    h = 150
    expected = np.array(
        [1, 0.2866666666666667, 0.85, 0.32666666666666666, 0.65, 0.013333333333333334, 0.8,
         -0.02, 0.5])
    normalized = r.normalized_params(w, h)
    assert np.array_equal(normalized, expected)


def test_opposite_rectangle_points_should_sum_up():
    r = Rectangle.random(w=100, h=100, alpha=1)
    acx = r.points[1][0] + r.points[3][0]
    bdx = r.points[0][0] + r.points[2][0]
    assert acx - 1 <= bdx <= acx + 1
    bdy = r.points[1][1] + r.points[3][1]
    acy = r.points[0][1] + r.points[2][1]
    assert bdy - 1 <= acy <= bdy + 1
