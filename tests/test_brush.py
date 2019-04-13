import numpy as np

from shapes.shape import Brush
from shapes.shape.shape import ARGS

CHANNELS = ARGS.channels


def test_params_intervals():
    w = 100
    h = 200
    color_intervals = [255, 255, 255] if CHANNELS == 3 else [255]
    expected = np.array([w, h, w, h, w, h, min(w, h) // 4])
    expected = np.append(expected, color_intervals)
    actual = Brush.params_intervals()(w=w, h=h)
    assert np.array_equal(expected, actual)
