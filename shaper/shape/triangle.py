import numpy as np
from numba import njit

from shaper.shape.shape import Shape, f


class Triangle(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha

    def __str__(self):
        return f'Triangle(A={self.points[0]}, B={self.points[1]}, C={self.points[2]})'

    @staticmethod
    def random(w, h, alpha):
        xs = np.random.randint(w, size=(3, 1))
        ys = np.random.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        assert points.shape == (3, 2), f'Shape of points: {points.shape}, expected: (3, 2)'
        return Triangle(points, alpha)

    @staticmethod
    def from_params(*params):
        return Triangle(points=np.array(params[:-1]).reshape(3, 2), alpha=params[-1])

    def get_bounds(self, h=None, w=None):
        return rasterize_triangle(self.points)

    def get_alpha(self):
        return self.alpha

    def args(self):
        return self.points.reshape(-1, ).astype(np.float64)

    @staticmethod
    def args_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h])


@njit("i8[:,:](i8[:,:])")
def rasterize_triangle(points):
    upper = np.argmin(points[:, 1:])
    lower = np.argmax(points[:, 1:])

    upper_point = points[upper]
    lower_point = points[lower]
    third_point = points[3 - (upper + lower)]

    bounds = np.empty((lower_point[1] - upper_point[1], 3), dtype=np.int64)

    i = 0
    for y in range(upper_point[1], lower_point[1]):
        start_x = f(upper_point[0], upper_point[1], lower_point[0], lower_point[1], y)

        if y < third_point[1]:
            x1, y1 = upper_point
            x2, y2 = third_point
        else:
            x1, y1 = third_point
            x2, y2 = lower_point

        end_x = f(x1, y1, x2, y2, y)
        bounds[i, 0] = start_x
        bounds[i, 1] = end_x
        bounds[i, 2] = y
        i += 1

    return bounds
