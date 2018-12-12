import numpy as np
import pytest

from shaper.imgutils import resize, mse_full, mse_partial


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

# todo: test avg_color when shape is straight line of 1px width
