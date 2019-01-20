import math

import numpy as np
from numba import njit

from shaper.shape.shape import Shape, merge_bounds
from shaper.util import timeit
from .triangle import rasterize_triangle


# todo: split to rectangle and quadrangle
class Rectangle(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha

    def __str__(self):
        return f'Rectangle(A={self.points[0]}, B={self.points[1]}, C={self.points[2]}, ' \
            f'D={self.points[3]})'

    @staticmethod
    @timeit
    def random(w, h, alpha, rng):
        def deg_to_rad(deg):
            return deg * math.pi / 180

        def rad_to_deg(rad):
            return rad * 180 / math.pi

        cx = rng.randint(w)
        cy = rng.randint(h)
        rw = rng.randint(1, w)
        rh = rng.randint(1, h)
        rot = rng.uniform(0, math.pi)

        x1 = cx + math.sin(
            (rot - deg_to_rad(90)) + math.asin(rh / (math.sqrt(rh * rh + rw * rw)))) * (
                 math.sqrt(rh * rh + rw * rw) / 2)
        y1 = cy - math.cos(
            (rot - deg_to_rad(90)) + math.asin(rh / (math.sqrt(rh * rh + rw * rw)))) * (
                 math.sqrt(rh * rh + rw * rw) / 2)
        x2 = x1 + math.cos(rot) * rw
        y2 = y1 + math.sin(rot) * rw
        x3 = x2 + math.cos(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        y3 = y2 + math.sin(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        x4 = x1 + math.cos(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        y4 = y1 + math.sin(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        return Rectangle(points=np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int64),
                         alpha=alpha)

    @staticmethod
    @timeit
    def from_params(*params):
        return Rectangle(points=np.array(params[:-1]).reshape(4, 2), alpha=params[-1])

    @staticmethod
    @timeit
    def from_normalized_params(w, h, *params):
        points = np.empty(shape=(4, 2))
        points[:, 0] = [int(x * w) for x in params[:-1][::2]]
        points[:, 1] = [int(y * h) for y in params[:-1][1::2]]
        return Rectangle(
            points=points,
            alpha=params[-1]
        )

    @timeit
    def get_bounds(self, h=None, w=None):
        return rasterize_quadrangle(self.points)

    def get_alpha(self):
        return self.alpha

    @timeit
    def params(self):
        return self.points.reshape(-1, ).astype(np.float64)

    @timeit
    def normalized_params(self, w, h):
        return np.append(
            self.points.reshape(-1, ).astype(np.float64) / np.array([w, h, w, h, w, h, w, h]),
            self.alpha)

    @staticmethod
    @timeit
    def params_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h, w, h])


@njit("i8[:,:](i8[:,:])")
def rasterize_quadrangle(points):
    bounds1 = rasterize_triangle(points[:-1])
    bounds2 = rasterize_triangle(points[1:])
    bounds1_len = bounds1.shape[0]
    bounds = np.empty(shape=(bounds1_len + bounds2.shape[0], 3), dtype=np.int64)
    bounds[:bounds1_len] = bounds1
    bounds[bounds1_len:] = bounds2
    return merge_bounds(bounds)
