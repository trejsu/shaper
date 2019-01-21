import math

import numpy as np
from numba import njit

from shaper.shape.shape import Shape, merge_bounds
from shaper.util import timeit
from .triangle import rasterize_triangle


class Quadrangle(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha

    def __str__(self):
        return f'Quadrangle(A={self.points[0]}, B={self.points[1]}, C={self.points[2]}, ' \
            f'D={self.points[3]})'

    @staticmethod
    def random(w, h, alpha, rng, scale=1):
        r = rng.randint(min(w, h) // 2)
        cx = rng.randint(w)
        cy = rng.randint(h)
        angles = np.sort(rng.uniform(0, 2 * math.pi, 4))
        xs = [cx + r * math.cos(angle) for angle in angles]
        ys = [cy + r * math.sin(angle) for angle in angles]
        points = np.concatenate((np.array(xs).reshape(4, 1), np.array(ys).reshape(4, 1)), axis=1)
        assert points.shape == (4, 2), f'Shape of points: {points.shape}, expected: (4, 2)'
        return Quadrangle(points, alpha)

    @staticmethod
    def from_params(*params):
        return Quadrangle(points=np.array(params[:-1]).reshape(4, 2), alpha=params[-1])

    @staticmethod
    def from_normalized_params(w, h, *params):
        points = np.empty(shape=(4, 2))
        points[:, 0] = [int(x * w) for x in params[:-1][::2]]
        points[:, 1] = [int(y * h) for y in params[:-1][1::2]]
        return Quadrangle(
            points=points,
            alpha=params[-1]
        )

    @staticmethod
    def params_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h, w, h])

    def get_bounds(self, h=None, w=None):
        return rasterize_quadrangle(self.points)

    def get_alpha(self):
        return self.alpha

    def params(self):
        return self.points.reshape(-1, ).astype(np.float64)

    def normalized_params(self, w, h):
        return np.append(
            self.points.reshape(-1, ).astype(np.float64) / np.array([w, h, w, h, w, h, w, h]),
            self.alpha
        )


class Rectangle(Shape):

    def __init__(self, cx, cy, w, h, rotation, alpha):
        self.cx = int(cx)
        self.cy = int(cy)
        self.w = max(1, int(w))
        self.h = max(1, int(h))
        self.rotation = rotation
        self.alpha = alpha

    def __str__(self):
        return f'Rectangle(cx={self.cx}, cy={self.cy}, w={self.w}, h={self.h}, rotation={self.rotation})'

    @staticmethod
    @timeit
    def random(w, h, alpha, rng, scale=1):
        cx = rng.randint(w)
        cy = rng.randint(h)
        w = rng.randint(1, w)
        h = rng.randint(1, h)
        rot = rng.uniform(0, math.pi)
        return Rectangle(cx=cx, cy=cy, w=w, h=h, rotation=rot, alpha=alpha)

    @staticmethod
    @timeit
    def from_params(*params):
        return Rectangle(*params)

    @staticmethod
    @timeit
    def from_normalized_params(w, h, *params):
        return Rectangle(
            cx=int(params[0] * w),
            cy=(params[1] * h),
            w=(params[2] * w),
            h=(params[3] * h),
            rotation=params[4] * math.pi,
            alpha=params[5]
        )

    @timeit
    def get_bounds(self, h=None, w=None):
        return rasterize_rectangle(self.cx, self.cy, self.w, self.h, self.rotation)

    def get_alpha(self):
        return self.alpha

    @timeit
    def params(self):
        return np.array([self.cx, self.cy, self.w, self.h, self.rotation], dtype=np.float64)

    @timeit
    def normalized_params(self, w, h):
        return np.append(
            np.array([self.cx, self.cy, self.w, self.h, self.rotation], dtype=np.float64) /
            np.array([w, h, w, h, math.pi]),
            self.alpha
        )

    @staticmethod
    @timeit
    def params_intervals():
        return lambda w, h: np.array([w, h, w - 1, h - 1, math.pi])


@njit("i8[:,:](i8[:,:])")
def rasterize_quadrangle(points):
    bounds1 = rasterize_triangle(points[:-1])
    triangle2_points = np.empty(shape=(3, 2), dtype=np.int64)
    triangle2_points[0] = points[0]
    triangle2_points[1] = points[2]
    triangle2_points[2] = points[3]
    bounds2 = rasterize_triangle(triangle2_points)
    bounds1_len = bounds1.shape[0]
    bounds = np.empty(shape=(bounds1_len + bounds2.shape[0], 3), dtype=np.int64)
    bounds[:bounds1_len] = bounds1
    bounds[bounds1_len:] = bounds2
    return merge_bounds(bounds)


@njit("f8(f8)")
def deg_to_rad(deg):
    return deg * math.pi / 180


@njit("f8(f8)")
def rad_to_deg(rad):
    return rad * 180 / math.pi


@njit("i8[:,:](i8,i8,i8,i8,f8)")
def rasterize_rectangle(cx, cy, w, h, rot):
    a = (rot - deg_to_rad(90)) + math.asin(h / (math.sqrt(h * h + w * w)))
    b = (math.sqrt(h * h + w * w) / 2)
    x1 = cx + math.sin(a) * b
    y1 = cy - math.cos(a) * b
    x2 = x1 + math.cos(rot) * w
    y2 = y1 + math.sin(rot) * w
    c = deg_to_rad(rad_to_deg(rot) + 90)
    x3 = x2 + math.cos(c) * h
    y3 = y2 + math.sin(c) * h
    x4 = x1 + math.cos(c) * h
    y4 = y1 + math.sin(c) * h
    points = np.empty(shape=(4, 2), dtype=np.int64)
    points[0] = x1, y1
    points[1] = x2, y2
    points[2] = x3, y3
    points[3] = x4, y4
    return rasterize_quadrangle(points)
