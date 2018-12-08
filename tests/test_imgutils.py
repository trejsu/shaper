import numpy as np
import pytest

from shaper.imgutils import resize


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
