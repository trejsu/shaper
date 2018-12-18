import numpy as np
from numba import njit

from .shape import Shape


# todo: add size
# todo: fix holes
class Curve(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha

    def __str__(self):
        return f'Curve: P0 = {self.points[0]}, P1 = {self.points[1]}, P2 = {self.points[2]}'

    @staticmethod
    def random(w, h, alpha):
        xs = np.random.randint(w, size=(3, 1))
        ys = np.random.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        return Curve(points, alpha)

    @staticmethod
    def from_params(*params):
        return Curve(points=np.array(params[:-1]).reshape(3, 2), alpha=params[-1])

    def get_bounds(self):
        return rasterize_curve(self.points)

    def get_alpha(self):
        return self.alpha

    def args(self):
        return self.points.reshape(-1, )

    @staticmethod
    def args_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h])


@njit("i8(f8, i8, i8, i8)")
def B(t, p0, p1, p2):
    return int((1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * p1 + t * t * p2)


@njit("i8[:,:](i8[:,:])")
def rasterize_curve(points):
    up = np.argmin(points[:, 1])
    down = np.argmax(points[:, 1])
    left = np.argmin(points[:, 0])
    right = np.argmax(points[:, 0])

    control_point = 1

    if control_point == up or control_point == down:
        ts = np.linspace(0, 1, points[right][0] - points[left][0] + 1)
        bounds = np.empty((points[right][0] - points[left][0] + 1, 3), dtype=np.int64)
    else:
        ts = np.linspace(0, 1, points[down][1] - points[up][1] + 1)
        bounds = np.empty((points[down][1] - points[up][1] + 1, 3), dtype=np.int64)

    i = 0
    for t in ts:
        x = B(t, points[0][0], points[1][0], points[2][0])
        y = B(t, points[0][1], points[1][1], points[2][1])

        bounds[i, 0] = x
        bounds[i, 1] = x
        bounds[i, 2] = y
        i += 1

    return bounds
