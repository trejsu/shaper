import numpy as np
import pytest

from shaper.imgutils import resize, mse_full


@pytest.mark.parametrize("input_w, input_h, w, h", [
    (100, 100, 100, 100),
    (100, 100, 50, 50),
    (100, 200, 50, 100),
    (100, 100, 50, 100),
    (100, 100, 100, 50)
])
def test_should_properly_resize_img(input_w, input_h, w, h):
    # given
    img = np.empty((input_h, input_w))

    # when
    result = resize(img, w, h)

    # then
    assert result.shape[0] == h
    assert result.shape[1] == w


def test_mse_full_should_be_zero_when_target_and_x_are_identical():
    # given
    target = np.random.random((100, 100))
    x = target.copy()

    # when
    mse = mse_full(target=target, x=x)

    # then
    assert mse == 0


def test_mse_full_should_be_max_when_target_is_white_and_x_is_black():
    # given
    target = np.full((100, 100), 255)
    x = np.zeros((100, 100))

    # when
    mse = mse_full(target=target, x=x)

    # then
    assert mse == 255 * 255


def test_mse_full_should_calculate_mse_properly():
    # given
    target = np.array([[2, 4], [4, 3]])
    x = np.array([[2, 5], [0, 2]])

    # when
    mse = mse_full(target=target, x=x)

    # then
    assert mse == 9 / 2
