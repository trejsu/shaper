import numpy as np
from numba import njit

from shaper.util import timeit
from .shape import Shape
from .shape import f


class Triangle(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha

    def __str__(self):
        return f'Triangle(A={self.points[0]}, B={self.points[1]}, C={self.points[2]})'

    @staticmethod
    @timeit
    def random(w, h, alpha, rng, scale=1):
        xs = rng.randint(w, size=(3, 1))
        ys = rng.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        assert points.shape == (3, 2), f'Shape of points: {points.shape}, expected: (3, 2)'
        center = np.sum(points, axis=0) / 3
        random_shift = [rng.randint(w), rng.randint(h)]
        points = scale * (points - center) + random_shift
        return Triangle(points, alpha)

    @staticmethod
    @timeit
    def from_params(*params):
        return Triangle(points=np.array(params[:-1]).reshape(3, 2), alpha=params[-1])

    @staticmethod
    @timeit
    def from_normalized_params(w, h, *params):
        points = np.empty(shape=(3, 2))
        points[:, 0] = [int(x * w) for x in params[:-1][::2]]
        points[:, 1] = [int(y * h) for y in params[:-1][1::2]]
        return Triangle(
            points=points,
            alpha=params[-1]
        )

    @timeit
    def get_bounds(self, h=None, w=None):
        return rasterize_triangle(self.points)

    def get_alpha(self):
        return self.alpha

    @timeit
    def params(self):
        return self.points.reshape(-1, ).astype(np.float64)

    @timeit
    def normalized_params(self, w, h):
        return np.append(
            arr=self.points.reshape(-1, ).astype(np.float64) / np.array([w, h, w, h, w, h]),
            values=self.alpha
        )

    @staticmethod
    @timeit
    def params_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h])


@njit("i8[:,:](i8[:,:])")
def rasterize_triangle(points):
    upper = np.argmin(points[:, 1])
    lower = np.argmax(points[:, 1])

    upper_point = points[upper]
    lower_point = points[lower]

    if upper == lower:
        bounds = np.empty((1, 3), dtype=np.int64)
        bounds[0, 0] = np.min(points[:, 0])
        bounds[0, 1] = np.max(points[:, 1])
        bounds[0, 2] = points[0, 1]
        return bounds

    third_point = points[3 - (upper + lower)]

    bounds = np.empty((lower_point[1] - upper_point[1] + 1, 3), dtype=np.int64)

    i = 0
    for y in range(upper_point[1], lower_point[1] + 1):
        start_x = f(upper_point[0], upper_point[1], lower_point[0], lower_point[1], y)

        if y < third_point[1]:
            x1, y1 = upper_point
            x2, y2 = third_point
        else:
            x1, y1 = third_point
            x2, y2 = lower_point

        if y1 == y2:
            bounds[i, 0] = start_x
            bounds[i, 1] = third_point[0]
        else:
            end_x = f(x1, y1, x2, y2, y)
            bounds[i, 0] = start_x
            bounds[i, 1] = end_x

        bounds[i, 2] = y
        i += 1

    return bounds
