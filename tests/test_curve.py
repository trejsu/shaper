import numpy as np

from shaper.shape.curve import Curve


def test_has_doubled_ys():
    c = Curve(points=np.array([[0, 0], [5, 10], [10, 5]]), alpha=1)
    c.get_bounds()
    assert c.has_doubled_ys()


def test_does_not_have_doubled_ys():
    c = Curve(points=np.array([[0, 0], [0, 4], [4, 10]]), alpha=1)
    c.get_bounds()
    assert not c.has_doubled_ys()
