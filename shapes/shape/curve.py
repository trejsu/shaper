import numpy as np
from numba import njit

from shapes.util import MIN_VALUE
from .shape import Shape


class Curve(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha

    def __str__(self):
        return f'Curve(P0={self.points[0]}, P1={self.points[1]}, P2={self.points[2]})'

    @staticmethod
    def random(w, h, alpha, rng, scale=1):
        xs = rng.randint(w, size=(3, 1))
        ys = rng.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        if scale != 1:
            center = np.sum(points, axis=0) / 3
            random_shift = [rng.randint(w), rng.randint(h)]
            points = scale * (points - center) + random_shift
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
        return rasterize_curve(self.points)

    def get_alpha(self):
        return self.alpha

    def params(self):
        return self.points.reshape(-1, ).astype(np.float64)

    def normalized_params(self, w, h):
        return np.append(
            arr=self.points.reshape(-1, ).astype(np.float64) / np.array([w, h, w, h, w, h]),
            values=self.alpha
        )

    @staticmethod
    def params_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h])

    def get_points(self, n):
        return get_points_on_curve(self.points, n)


@njit("i8(f8, i8, i8, i8)")
def bezier(t, p0, p1, p2):
    if p0 == p1 == p2:
        return p0
    return int((1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * p1 + t * t * p2)


@njit
def extremum(p0, p1, p2):
    a = p0 - p1
    b = p0 - 2 * p1 + p2

    if a == 0 or np.sign(a) != np.sign(b) or b == 0 or abs(a) > abs(b):
        return p0
    return bezier(a / b, p0, p1, p2)


@njit("i8[:,:](i8[:,:])")
def rasterize_curve(points):
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
            bounds[i - 1, 0] = min(bounds[i - 1, 0], x)
            bounds[i - 1, 1] = max(bounds[i - 1, 1], x)
        else:
            bounds[i, 0] = x
            bounds[i, 1] = x
            bounds[i, 2] = y
            i += 1

        prev_y = y

    return bounds[0:i]


@njit("i8[:,:](i8[:,:], i8)")
def get_points_on_curve(control_points, n):
    x0, y0 = control_points[0]
    x1, y1 = control_points[1]
    x2, y2 = control_points[2]

    ts = np.linspace(0, 1, n)
    points = np.empty((n, 2), dtype=np.int64)

    i = 0
    for t in ts:
        points[i, 0] = bezier(t, x0, x1, x2)
        points[i, 1] = bezier(t, y0, y1, y2)
        i += 1

    return points
