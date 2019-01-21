from abc import abstractmethod

import numpy as np
from numba import njit

from shaper.util import timeit, MIN_VALUE


class Shape(object):
    @staticmethod
    @abstractmethod
    def random(w, h, alpha, rng, scale):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_params(*params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_normalized_params(w, h, *params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def params_intervals():
        raise NotImplementedError

    @abstractmethod
    def get_bounds(self, h, w):
        raise NotImplementedError

    @abstractmethod
    def get_alpha(self):
        raise NotImplementedError

    @abstractmethod
    def params(self):
        raise NotImplementedError

    @abstractmethod
    def normalized_params(self, w, h):
        raise NotImplementedError

    @timeit
    def render(self, img, target):
        bounds = self.get_bounds(h=img.shape[0], w=img.shape[1])
        crop_bounds(bounds=bounds, h=img.shape[0], w=img.shape[1])
        color = average_color(img=target, bounds=bounds)
        alpha = self.get_alpha()
        assert 0 <= alpha <= 1, f'alpha out of bounds = {alpha}'
        render(img=img, bounds=bounds, color=color, alpha=alpha)
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
        for x in range(min(bounds[i, 0], bounds[i, 1]), max(bounds[i, 0], bounds[i, 1]) + 1):
            img[bounds[i, 2], x, 0] = img[bounds[i, 2], x, 0] * a_current + a_r
            img[bounds[i, 2], x, 1] = img[bounds[i, 2], x, 1] * a_current + a_g
            img[bounds[i, 2], x, 2] = img[bounds[i, 2], x, 2] * a_current + a_b


@njit("Tuple((i8, i8, i8))(f8[:,:,:], i8[:,:])")
def average_color(img, bounds):
    r, g, b, pixels = 0, 0, 0, 1e-10

    for i in range(len(bounds)):
        for x in range(min(bounds[i, 0], bounds[i, 1]), max(bounds[i, 0], bounds[i, 1]) + 1):
            r += img[bounds[i, 2], x, 0]
            g += img[bounds[i, 2], x, 1]
            b += img[bounds[i, 2], x, 2]
            pixels += 1

    return r // pixels, g // pixels, b // pixels


@njit("(i8[:,:], i8, i8)")
def crop_bounds(bounds, h, w):
    for i in range(len(bounds)):
        bounds[i, 0] = max(0, min(w - 1, bounds[i, 0]))
        bounds[i, 1] = max(0, min(w - 1, bounds[i, 1]))
        bounds[i, 2] = max(0, min(h - 1, bounds[i, 2]))


@njit("i8[:,:](i8[:,:])")
def merge_bounds(bounds):
    min_y = np.min(bounds[:, 2])
    max_y = np.max(bounds[:, 2])

    num_y = max_y - min_y + 1

    merged = np.full(shape=(num_y, 3), fill_value=MIN_VALUE, dtype=np.int64)

    merged[:, 2] = np.arange(min_y, max_y + 1)

    for i in range(len(bounds)):
        start, end, y = bounds[i]
        j = y - min_y
        current = merged[j]
        if current[0] == MIN_VALUE:
            merged[j, 0] = min(start, end)
            merged[j, 1] = max(start, end)
        else:
            merged[j, 0] = min(start, end, current[0])
            merged[j, 1] = max(start, end, current[1])

    return merged
