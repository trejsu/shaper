import os

import numpy as np
import pytest

from config import ROOT_DIR
from shapes.shape import Quadrangle
from shapes.util import resize, l2_full, update_l2, normalize, bounds_to_pixels, read_img, hex_to_rgb


@pytest.mark.parametrize("input_w, input_h, w, h", [
    (100, 100, 100, 100),
    (100, 100, 50, 50),
    (100, 200, 50, 100),
    (100, 100, 50, 100),
    (100, 100, 100, 50)
])
def test_should_properly_resize_img(input_w, input_h, w, h):
    img = np.empty((input_h, input_w))
    result = resize(img, w, h)
    assert result.shape[0] == h
    assert result.shape[1] == w


def test_l2_full_should_be_all_zeros_when_target_and_x_are_identical():
    target = np.random.random((100, 100))
    x = target.copy()
    l2 = l2_full(target=target, x=x)
    assert np.array_equal(l2, np.zeros(target.shape))


def test_l2_full_should_be_max_when_target_is_white_and_x_is_black():
    target = np.full((100, 100), 255)
    x = np.zeros((100, 100))
    l2 = l2_full(target=target, x=x)
    assert np.array_equal(l2, np.full(target.shape, 255 * 255))


def test_l2_full_should_calculate_l2_properly():
    target = np.array([[2, 4], [4, 3]])
    x = np.array([[2, 5], [0, 2]])
    l2 = l2_full(target=target, x=x)
    assert np.array_equal(l2, np.array([[0, 1], [16, 1]]))


def test_l2_full_should_return_float_array():
    target = np.random.random((100, 100))
    x = np.random.random((100, 100))
    l2 = l2_full(target=target, x=x)
    assert l2.dtype == np.float


def test_update_l2():
    l2 = np.zeros((100, 100, 3))
    bounds = np.array([[50, 60, 50], [50, 60, 51], [50, 60, 52]])
    img = np.random.random((100, 100, 3))
    target = np.random.random((100, 100, 3))
    update_l2(l2, bounds, img, target)
    assert np.all(l2[50, 50:61])
    assert np.all(l2[51, 50:61])
    assert np.all(l2[52, 50:61])


def test_update_l2_should_calculate_the_same_values_as_full_l2_for_bounded_area():
    l2 = np.zeros((100, 100, 3))
    bounds = np.array([[50, 60, 50], [50, 60, 51], [50, 60, 52]])
    img = np.random.random((100, 100, 3))
    target = np.random.random((100, 100, 3))
    update_l2(l2, bounds, img, target)
    full = l2_full(target, img)
    assert np.array_equal(full[np.where(l2 > 0)], l2[np.where(l2 > 0)])


def test_update_l2_should_have_the_same_effect_as_full_l2():
    bounds = np.array([[50, 60, 50], [50, 60, 51], [50, 60, 52]])
    target = np.random.random((100, 100, 3))
    img = np.random.random((100, 100, 3))
    l2 = l2_full(target, img)
    img[50, 50:61] = 0.5
    img[51, 50:61] = 0.4
    img[52, 50:61] = 0.3
    update_l2(l2, bounds, img, target)
    assert np.array_equal(l2, l2_full(target, img))


@pytest.mark.parametrize("x1, y1, x2, y2, x3, y3, x4, y4", [
    (244, 93, 509, 125, 449, 628, 184, 596),
    (140, -237, 421, 76, 47, 409, -233, 95),
    (427, 145, 139, 436, 108, 406, 396, 115)
])
def test_broken_partial_l2(x1, y1, x2, y2, x3, y3, x4, y4):
    img = np.zeros((500, 500, 3))
    target = np.zeros((500, 500, 3))
    distance = np.zeros((500, 500, 3))
    assert img.dtype == np.float
    assert target.dtype == np.float
    assert distance.dtype == np.float
    r = Quadrangle(
        points=np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int64),
        alpha=0.5
    )
    bounds = r.render(img, target, None)
    update_l2(distance=distance, bounds=bounds, img=img, target=target)
    assert np.array_equal(distance, l2_full(target, img))
    assert np.average(distance) == np.average(l2_full(target, img))


def test_normalize_should_not_return_nans_when_array_has_the_same_elements():
    arr = np.full(shape=(100,), fill_value=33)
    normalized = normalize(arr)
    assert not np.any(np.isnan(normalized))


def test_normalize_should_return_the_same_values_when_array_has_the_same_elements():
    arr = np.full(shape=(100,), fill_value=33)
    normalized = normalize(arr)
    assert np.all(normalized[0] == normalized)


def test_bounds_to_pixels():
    bounds = np.array([[3, 10, 10], [3, 11, 11], [4, 12, 12]])
    expected_pixels = np.array(
        [[3, 10], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10], [9, 10], [10, 10],
         [3, 11], [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [9, 11], [10, 11],
         [11, 11], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12], [10, 12],
         [11, 12], [12, 12]]
    )
    pixels = bounds_to_pixels(bounds)
    assert np.array_equal(pixels, expected_pixels)


def test_bounds_to_pixels_start_x_bigger_than_end_x():
    bounds = np.array([[10, 3, 10], [11, 3, 11], [12, 4, 12]])
    expected_pixels = np.array(
        [[3, 10], [4, 10], [5, 10], [6, 10], [7, 10], [8, 10], [9, 10], [10, 10],
         [3, 11], [4, 11], [5, 11], [6, 11], [7, 11], [8, 11], [9, 11], [10, 11],
         [11, 11], [4, 12], [5, 12], [6, 12], [7, 12], [8, 12], [9, 12], [10, 12],
         [11, 12], [12, 12]]
    )
    pixels = bounds_to_pixels(bounds)
    assert np.array_equal(pixels, expected_pixels)


def test_read_img_jpg():
    path = os.path.join(ROOT_DIR, 'data/tree/tree.jpg')
    img = read_img(path)
    assert img.shape[2] == 3
    assert np.any(img > 1)


def test_read_img_png():
    path = os.path.join(ROOT_DIR, 'data/mnist/mnist-1.png')
    img = read_img(path)
    assert img.shape[2] == 3
    assert np.any(img > 1)


@pytest.mark.parametrize("hex, r, g, b", [
    ('ffffff', 255, 255, 255),
    ('000000', 0, 0, 0),
    ('123456', 18, 52, 86),
    ('abcdef', 171, 205, 239)
])
def test_hex_to_rgb(hex, r, g, b):
    rgb = hex_to_rgb(hex)
    assert rgb[0] == r
    assert rgb[1] == g
    assert rgb[2] == b
