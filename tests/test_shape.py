import numpy as np
import pytest

from shapes.shape import Triangle, Rectangle, QuadrangleBrush, Quadrangle, Ellipse, Curve, \
    from_index, index_of
from shapes.shape.shape import crop_bounds


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


@pytest.mark.parametrize("index, shape_cls", [
    (0, Triangle),
    (1, Rectangle),
    (2, Ellipse),
    (3, Quadrangle),
    (4, QuadrangleBrush),
    (5, Curve)
])
def test_from_index_should_return_proper_shape_class_given_index(index, shape_cls):
    cls = from_index(index)
    assert cls == shape_cls


@pytest.mark.parametrize("shape_cls, index", [
    (Triangle, 0),
    (Rectangle, 1),
    (Ellipse, 2),
    (Quadrangle, 3),
    (QuadrangleBrush, 4),
    (Curve, 5)
])
def test_index_of_should_return_proper_index_given_shape_class(shape_cls, index):
    idx = index_of(shape_cls)
    assert idx == index


@pytest.mark.parametrize("shape_cls, expected_params_len", [
    (Triangle, 8),
    (Rectangle, 7),
    (Ellipse, 7),
    (Quadrangle, 10),
    (QuadrangleBrush, 9),
    (Curve, 8)
])
def test_params_len_should_return_proper_value_for_each_shape_class(shape_cls, expected_params_len):
    params_len = shape_cls.params_len()
    assert params_len == expected_params_len
