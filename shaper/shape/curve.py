import numpy as np
from numba import njit

from shaper.util import MIN_VALUE
from .shape import Shape


class Curve(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha
        self.extremum = None
        self.num_bounds = None

    def __str__(self):
        return f'Curve(P0={self.points[0]}, P1={self.points[1]}, P2={self.points[2]})'

    @staticmethod
    def random(w, h, alpha):
        xs = np.random.randint(w, size=(3, 1))
        ys = np.random.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        return Curve(points, alpha)

    @staticmethod
    def from_params(*params):
        return Curve(points=np.array(params[:-1]).reshape(3, 2), alpha=params[-1])

    @staticmethod
    def from_normalized_params(w, h, *params):
        points = np.empty(shape=(3, 2))
        points[:, 0] = [int(x * w) for x in params[:-1][::2]]
        points[:, 1] = [int(y * h) for y in params[:-1][1::2]]
        return Curve(
            points=points,
            alpha=params[-1]
        )

    def get_bounds(self, h=None, w=None):
        bounds, self.extremum = rasterize_curve(self.points, 0)
        self.num_bounds = bounds.shape[0]
        return bounds

    def get_alpha(self):
        return self.alpha

    def params(self):
        return self.points.reshape(-1, ).astype(np.float64)

    def normalized_params(self, w, h):
        return self.points.reshape(-1, ).astype(np.float64) / np.array([w, h, w, h, w, h])

    @staticmethod
    def params_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h])

    def has_doubled_ys(self):
        if self.extremum is None:
            raise Exception('Extremum value is not initialized. Call get_bounds first.')
        return self.extremum != self.points[0][1] and self.extremum != self.points[2][1]


@njit("i8(f8, i8, i8, i8)")
def bezier(t, p0, p1, p2):
    if p0 == p1 == p2:
        return p0
    return int((1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * p1 + t * t * p2)


@njit
def extremum(p0, p1, p2):
    a = p0 - p1
    b = p0 - 2 * p1 + p2

    if a == 0:
        return p0
    if np.sign(a) != np.sign(b):
        return p0
    if b == 0:
        return p0
    if abs(a) > abs(b):
        return p1
    return bezier(a / b, p0, p1, p2)


@njit("Tuple((i8[:,:], i8))(i8[:,:], i8)")
def rasterize_curve(points, size):
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]

    ext = extremum(y0, y1, y2)

    max_num_t = 2 * max(abs(x0 - x1), abs(x1 - x2), abs(y0 - y1), abs(y1 - y2)) + 1
    ts = np.linspace(0, 1, max_num_t)

    num_bounds = abs(y0 - ext) + abs(y2 - ext) + 1

    bounds = np.empty((num_bounds, 3), dtype=np.int64)

    i = 0
    prev_y = MIN_VALUE
    for t in ts:

        x = bezier(t, x0, x1, x2)
        y = bezier(t, y0, y1, y2)

        if y == prev_y:
            bounds[i - 1, 0] = min(bounds[i - 1, 0], x - size)
            bounds[i - 1, 1] = max(bounds[i - 1, 1], x + size)
        else:
            bounds[i, 0] = x - size
            bounds[i, 1] = x + size
            bounds[i, 2] = y
            i += 1

        prev_y = y

    return bounds[0:i], ext
