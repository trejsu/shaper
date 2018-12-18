import numpy as np
import pytest

from shaper.shape.ellipse import rasterize_ellipse, Ellipse


def test_bounds_for_ellipse_should_have_proper_limits():
    bounds = rasterize_ellipse(a=5, b=3, h=2, k=5, r=2)
    upper_y = bounds[np.argmin(bounds[:, 2])][2]
    lower_y = bounds[np.argmax(bounds[:, 2])][2]
    assert upper_y == 0, f'Actual upper_y: {upper_y}'
    assert lower_y == 9, f'Actual lower_y: {lower_y}'


def test_bounds_for_ellipse_should_be_properly_calculated():
    expected_bounds = np.array(
        [[0, 0, 0], [0, 2, 1], [-1, 3, 2], [-1, 4, 3], [-1, 4, 4], [-1, 5, 5], [0, 5, 6], [0, 5, 7],
         [0, 5, 8], [1, 4, 9]])
    bounds = rasterize_ellipse(a=5, b=3, h=2, k=5, r=2)
    assert np.array_equal(bounds, expected_bounds)


@pytest.mark.parametrize("a, b", [
    (-432532, 43),
    (32, -342),
    (32, 54),
    (-5423542, -432),
    (0, 43),
    (32, 0),
    (0, 0),
    (0.5423, 43),
    (32, 0.7562),
    (0, 0.99999),
])
def test_from_params_should_not_produce_ellipse_with_a_and_b_less_than_1(a, b):
    e = Ellipse.from_params(a, b, 1, 2, 3, 4)
    assert e.a > 0
    assert e.b > 0
