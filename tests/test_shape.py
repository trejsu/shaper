import numpy as np
import pytest

from shaper.shape import crop_bounds, Rectangle, Ellipse


def test_should_leave_bounds_unchanged_if_no_need_to_crop():
    bounds = np.array([[50, 60, 50], [50, 60, 51], [50, 60, 52]])
    bounds_to_crop = bounds.copy()
    w = 100
    h = 100
    crop_bounds(bounds=bounds_to_crop, w=w, h=h)
    assert np.array_equal(bounds, bounds_to_crop)


def test_should_crop_negative_x1s_to_zeros():
    bounds = np.array([[-1, 60, 50], [50, 60, 51], [50, 60, 52]])
    w = 100
    h = 100
    crop_bounds(bounds=bounds, w=w, h=h)
    assert np.all(bounds >= 0)


def test_should_crop_too_big_x2s_to_max_width():
    bounds = np.array([[50, 60, 50], [50, 100, 51], [50, 60, 52]])
    w = 100
    h = 100
    crop_bounds(bounds=bounds, w=w, h=h)
    assert np.all(bounds < w)


def test_should_crop_negative_ys_to_zeros():
    bounds = np.array([[50, 60, -1], [50, 100, 51], [50, 60, 52]])
    w = 100
    h = 100
    crop_bounds(bounds=bounds, w=w, h=h)
    assert np.all(bounds >= 0)


def test_should_crop_too_big_ys_to_max_height():
    bounds = np.array([[50, 60, 50], [50, 100, 51], [50, 60, 100]])
    w = 100
    h = 100
    crop_bounds(bounds=bounds, w=w, h=h)
    assert np.all(bounds < h)


@pytest.mark.parametrize("x1, x2, y, cropped_x1, cropped_x2, cropped_y", [
    (-1, -2, 50, 0, 0, 50),
    (-1, 150, 50, 0, 99, 50),
    (140, 150, 50, 99, 99, 50),
    (-1, 24, -30, 0, 24, 0),
    (50, 150, 150, 50, 99, 99),
])
def test_should_crop_when_multiple_variables_are_out_of_bounds(x1, x2, y, cropped_x1, cropped_x2,
    cropped_y):
    bounds = np.array([[x1, x2, y]])
    crop_bounds(bounds=bounds, w=100, h=100)
    assert bounds[0][0] == cropped_x1, f'actual x1: {bounds[0][0]}, expected x1: {cropped_x1}'
    assert bounds[0][1] == cropped_x2, f'actual x2: {bounds[0][1]}, expected x2: {cropped_x2}'
    assert bounds[0][2] == cropped_y, f'actual y: {bounds[0][2]}, expected y: {cropped_y}'


def test_opposite_rectangle_points_should_sum_up():
    r = Rectangle.random(w=100, h=100, alpha=1)
    acx = r.points[1][0] + r.points[3][0]
    bdx = r.points[0][0] + r.points[2][0]
    assert acx - 1 <= bdx <= acx + 1
    bdy = r.points[1][1] + r.points[3][1]
    acy = r.points[0][1] + r.points[2][1]
    assert bdy - 1 <= acy <= bdy + 1


def test_ellipse_coefficients_should_be_properly_calculated():
    cx = 2
    cy = 5
    a = 5
    b = 3
    rotation = 2
    e = Ellipse.from_params(**{'cx': cx, 'cy': cy, 'a': a, 'b': b, 'rotation': rotation})
    A = 22.22
    B = 11.77
    C = -28.37
    D = -93.49
    E = -12.10
    F = 37.09
    np.testing.assert_almost_equal(e.A, A, decimal=2)
    np.testing.assert_almost_equal(e.B, B, decimal=2)
    np.testing.assert_almost_equal(e.C, C, decimal=2)
    np.testing.assert_almost_equal(e.D, D, decimal=2)
    np.testing.assert_almost_equal(e.E, E, decimal=2)
    np.testing.assert_almost_equal(e.F, F, decimal=2)


# todo: remove?
def test_calculating_ellipse_limits():
    e = Ellipse.from_params(**{'cx': 2, 'cy': 5, 'a': 5, 'b': 3, 'rotation': 2})
    a = 4 * e.A * e.B - (e.E * e.E)
    b = 4 * e.A * e.D - 2 * e.C * e.E
    c = 4 * e.A * e.F - (e.C * e.C)
    roots = np.roots([a, b, c])
    np.testing.assert_almost_equal(np.real(roots[0]), 9.715, decimal=3)
    np.testing.assert_almost_equal(np.real(roots[1]), 0.285, decimal=3)


# todo: remove?
@pytest.mark.parametrize("y, expected_x1, expected_x2", [
    (9.715, 3.284, 3.284),
    (7.525, 0.001, 5.374),
    (4, -1.381, 4.837)
])
def test_calculating_x1_and_x2_given_y_for_ellipse(y, expected_x1, expected_x2):
    e = Ellipse.from_params(**{'cx': 2, 'cy': 5, 'a': 5, 'b': 3, 'rotation': 2})
    a = e.A
    b = e.C + e.E * y
    c = e.B * y * y + e.D * y + e.F
    roots = np.roots([a, b, c])
    np.testing.assert_almost_equal(np.real(roots[1]), expected_x1, decimal=3)
    np.testing.assert_almost_equal(np.real(roots[0]), expected_x2, decimal=3)
