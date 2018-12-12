from abc import abstractmethod

import numpy as np
from numba import njit


class Shape(object):
    @staticmethod
    @abstractmethod
    def random(w, h, alpha):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_params(**params):
        raise NotImplementedError

    @abstractmethod
    def render(self, img, target):
        raise NotImplementedError


class Triangle(Shape):

    def __init__(self, points, alpha):
        self.points = points
        self.alpha = alpha

    def __str__(self):
        return f'Triangle: A = {self.points[0]}, B = {self.points[1]}, C = {self.points[2]}'

    @staticmethod
    def random(w, h, alpha):
        xs = np.random.randint(w, size=(3, 1))
        ys = np.random.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        assert points.shape == (3, 2), f'Shape of points: {points.shape}, expected: (3, 2)'
        return Triangle(points, alpha)

    @staticmethod
    def from_params(**params):
        pass

    def render(self, img, target):
        bounds = rasterize_triangle(self.points)
        color = average_color(img=target, bounds=bounds)
        render(img=img, bounds=bounds, color=color, alpha=self.alpha)
        return bounds


@njit("f8(i8, i8, i8, i8, i8)")
def f(x1, y1, x2, y2, y):
    return ((y - y1) * x2 + (y2 - y) * x1) // (y2 - y1)


@njit("(f8[:,:,:], i8[:,:], Tuple((i8, i8, i8)), f8)")
def render(img, bounds, color, alpha):
    a_current = 1 - alpha
    r, g, b = color
    a_r = r * alpha
    a_g = g * alpha
    a_b = b * alpha

    for i in range(len(bounds)):
        for x in range(bounds[i, 0], bounds[i, 1] + 1, 1 if bounds[i, 0] < bounds[i, 1] else -1):
            img[bounds[i, 2], x, 0] = img[bounds[i, 2], x, 0] * a_current + a_r
            img[bounds[i, 2], x, 1] = img[bounds[i, 2], x, 1] * a_current + a_g
            img[bounds[i, 2], x, 2] = img[bounds[i, 2], x, 2] * a_current + a_b


@njit("Tuple((i8, i8, i8))(f8[:,:,:], i8[:,:])")
def average_color(img, bounds):
    r, g, b, pixels = 0, 0, 0, 1e-10

    for i in range(len(bounds)):
        for x in range(bounds[i, 0], bounds[i, 1] + 1, 1 if bounds[i, 0] < bounds[i, 1] else -1):
            r += img[bounds[i, 2], x, 0]
            g += img[bounds[i, 2], x, 1]
            b += img[bounds[i, 2], x, 2]
            pixels += 1

    color = r // pixels, g // pixels, b // pixels
    return color


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
