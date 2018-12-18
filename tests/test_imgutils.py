import numpy as np
import pytest

from shaper.imgutils import resize, mse_full, mse_partial, update_mse
from shaper.shape.rectangle import Rectangle


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


def test_mse_full_should_be_all_zeros_when_target_and_x_are_identical():
    target = np.random.random((100, 100))
    x = target.copy()
    mse = mse_full(target=target, x=x)
    assert np.array_equal(mse, np.zeros(target.shape))


def test_mse_full_should_be_max_when_target_is_white_and_x_is_black():
    target = np.full((100, 100), 255)
    x = np.zeros((100, 100))
    mse = mse_full(target=target, x=x)
    assert np.array_equal(mse, np.full(target.shape, 255 * 255))


def test_mse_full_should_calculate_mse_properly():
    target = np.array([[2, 4], [4, 3]])
    x = np.array([[2, 5], [0, 2]])
    mse = mse_full(target=target, x=x)
    assert np.array_equal(mse, np.array([[0, 1], [16, 1]]))


def test_mse_partial_with_full_mask_should_be_equal_to_mse_full():
    target = np.random.random((100, 100))
    x = np.random.random((100, 100))
    mask = np.ones(target.shape)
    partial = mse_partial(target=target, x=x, mask=mask)
    full = mse_full(target=target, x=x)
    assert np.array_equal(partial, full)


def test_mse_partial_with_empty_mask_should_be_all_zeros():
    target = np.random.random((100, 100))
    x = np.random.random((100, 100))
    mask = np.zeros(target.shape)
    mse = mse_partial(target=target, x=x, mask=mask)
    assert np.all(mse == 0)


def test_mse_partial_should_calculate_result_for_proper_elements():
    target = np.random.random((100, 100))
    x = np.random.random((100, 100))
    mask = np.random.randint(2, size=target.shape)
    mse = mse_partial(target=target, x=x, mask=mask)
    assert np.all(mse[np.where(mask == 0)] == 0)
    assert np.all(mse[np.where(mask == 1)] != 0)


def test_mse_partial_should_calculate_mse_properly():
    target = np.array([[2, 4], [4, 3]])
    x = np.array([[2, 5], [0, 2]])
    mask = np.array([[1, 0], [0, 1]])
    mse = mse_partial(target=target, x=x, mask=mask)
    assert np.array_equal(mse, np.array([[0, 0], [0, 1]]))


def test_mse_full_should_return_float_array():
    target = np.random.random((100, 100))
    x = np.random.random((100, 100))
    mse = mse_full(target=target, x=x)
    assert mse.dtype == np.float


def test_mse_partial_should_return_float_array():
    target = np.random.random((100, 100))
    x = np.random.random((100, 100))
    mask = np.random.randint(2, size=target.shape)
    mse = mse_partial(target=target, x=x, mask=mask)
    assert mse.dtype == np.float


def test_update_mse():
    mse = np.zeros((100, 100, 3))
    bounds = np.array([[50, 60, 50], [50, 60, 51], [50, 60, 52]])
    img = np.random.random((100, 100, 3))
    target = np.random.random((100, 100, 3))
    update_mse(mse, bounds, img, target)
    assert np.all(mse[50, 50:61])
    assert np.all(mse[51, 50:61])
    assert np.all(mse[52, 50:61])


def test_update_mse_should_calculate_the_same_values_as_full_mse_for_bounded_area():
    mse = np.zeros((100, 100, 3))
    bounds = np.array([[50, 60, 50], [50, 60, 51], [50, 60, 52]])
    img = np.random.random((100, 100, 3))
    target = np.random.random((100, 100, 3))
    update_mse(mse, bounds, img, target)
    full = mse_full(target, img)
    assert np.array_equal(full[np.where(mse > 0)], mse[np.where(mse > 0)])


def test_update_mse_should_have_the_same_effect_as_full_mse():
    bounds = np.array([[50, 60, 50], [50, 60, 51], [50, 60, 52]])
    target = np.random.random((100, 100, 3))
    img = np.random.random((100, 100, 3))
    mse = mse_full(target, img)
    img[50, 50:61] = 0.5
    img[51, 50:61] = 0.4
    img[52, 50:61] = 0.3
    update_mse(mse, bounds, img, target)
    assert np.array_equal(mse, mse_full(target, img))


@pytest.mark.parametrize("x1, y1, x2, y2, x3, y3, x4, y4", [
    (244, 93, 509, 125, 449, 628, 184, 596),
    (140, -237, 421, 76, 47, 409, -233, 95),
    (427, 145, 139, 436, 108, 406, 396, 115)
])
def test_broken_partial_mse(x1, y1, x2, y2, x3, y3, x4, y4):
    img = np.zeros((500, 500, 3))
    target = np.zeros((500, 500, 3))
    mse = np.zeros((500, 500, 3))
    assert img.dtype == np.float
    assert target.dtype == np.float
    assert mse.dtype == np.float
    r = Rectangle(points=np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int64),
                  alpha=0.5)
    bounds = r.render(img, target)
    update_mse(mse=mse, bounds=bounds, img=img, target=target)
    assert np.array_equal(mse, mse_full(target, img))
    assert np.average(mse) == np.average(mse_full(target, img))
