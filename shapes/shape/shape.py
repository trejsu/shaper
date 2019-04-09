from abc import abstractmethod

import numpy as np
from numba import njit

from shapes.util import MIN_VALUE


class Shape(object):

    def __init__(self):
        self.__color = None

    @property
    def color(self):
        return self.__color

    @color.setter
    def color(self, c):
        self.__color = np.clip(c, 0, 255)

    @classmethod
    def random(cls, w, h, alpha, rng, scale):
        shape = cls._random(w, h, alpha, rng, scale)
        shape.color = Shape.random_color(rng)
        return shape

    @staticmethod
    @abstractmethod
    def _random(w, h, alpha, rng, scale):
        raise NotImplementedError

    @classmethod
    def from_params(cls, *params):
        shape = cls._from_params(*params[:-4], params[-1])
        shape.color = params[-4:-1]
        return shape

    @staticmethod
    @abstractmethod
    def _from_params(*params):
        raise NotImplementedError

    @classmethod
    def from_normalized_params(cls, w, h, *params):
        shape = cls._from_normalized_params(w, h, *params[:-4], params[-1])
        shape.color = params[-4:-1]
        return shape

    @staticmethod
    @abstractmethod
    def _from_normalized_params(w, h, *params):
        raise NotImplementedError

    @classmethod
    def params_intervals(cls):
        intervals = cls._params_intervals()
        return lambda w, h: np.append(intervals(w, h), [255, 255, 255])

    @staticmethod
    @abstractmethod
    def _params_intervals():
        raise NotImplementedError

    @abstractmethod
    def get_bounds(self, h, w):
        raise NotImplementedError

    @abstractmethod
    def get_alpha(self):
        raise NotImplementedError

    def params(self):
        return np.append(self._params(), self.color_expanded())

    @abstractmethod
    def _params(self):
        raise NotImplementedError

    def normalized_params(self, w, h):
        params = self._normalized_params(w, h)
        return np.append(np.append(params[:-1], self.color_expanded()), params[-1])

    def color_expanded(self):
        return [self.color[0], self.color[1], self.color[2]]

    @abstractmethod
    def _normalized_params(self, w, h):
        raise NotImplementedError

    def render(self, img, target):
        bounds = self.get_bounds(h=img.shape[0], w=img.shape[1])
        crop_bounds(bounds=bounds, h=img.shape[0], w=img.shape[1])
        color = self.resolve_color(bounds, target)
        alpha = self.get_alpha()
        assert 0 <= alpha <= 1, f'alpha out of bounds = {alpha}'
        if img.shape[-1] == 1:
            if isinstance(color, np.float64):
                color = color.astype(np.int64)
            assert isinstance(color, np.int64), f'color = {color}, type(color) = {type(color)}'
            render_1_channel(img=img, bounds=bounds, color=color, alpha=alpha)
        else:
            assert isinstance(color, tuple), f'color = {color}, type(color) = {type(color)}'
            render_3_channels(img=img, bounds=bounds, color=color, alpha=alpha)
        return bounds

    def resolve_color(self, bounds, target):
        if self.color is None:
            if target.shape[-1] == 1:
                return average_color_1_channel(img=target, bounds=bounds)
            else:
                return average_color_3_channels(img=target, bounds=bounds)
        if target.shape[-1] == 1:
            return self.color[0]
        if not isinstance(self.color, tuple):
            return tuple(self.color)
        else:
            return self.color

    @staticmethod
    def random_color(rng):
        return rng.randint(0, 255, (3,))


@njit("f8(i8, i8, i8, i8, i8)")
def f(x1, y1, x2, y2, y):
    return ((y - y1) * x2 + (y2 - y) * x1) // (y2 - y1)


@njit("(f8[:,:,:], i8[:,:], Tuple((i8, i8, i8)), f8)")
def render_3_channels(img, bounds, color, alpha):
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


@njit("(f8[:,:,:], i8[:,:], i8, f8)")
def render_1_channel(img, bounds, color, alpha):
    a_current = 1 - alpha
    a_c = color * alpha

    for i in range(len(bounds)):
        for x in range(min(bounds[i, 0], bounds[i, 1]), max(bounds[i, 0], bounds[i, 1]) + 1):
            img[bounds[i, 2], x, 0] = img[bounds[i, 2], x, 0] * a_current + a_c


@njit("Tuple((i8, i8, i8))(f8[:,:,:], i8[:,:])")
def average_color_3_channels(img, bounds):
    r, g, b, pixels = 0, 0, 0, 1e-10

    for i in range(len(bounds)):
        for x in range(min(bounds[i, 0], bounds[i, 1]), max(bounds[i, 0], bounds[i, 1]) + 1):
            r += img[bounds[i, 2], x, 0]
            g += img[bounds[i, 2], x, 1]
            b += img[bounds[i, 2], x, 2]
            pixels += 1

    return r // pixels, g // pixels, b // pixels


@njit("i8(f8[:,:,:], i8[:,:])")
def average_color_1_channel(img, bounds):
    color, pixels = 0, 1e-10

    for i in range(len(bounds)):
        for x in range(min(bounds[i, 0], bounds[i, 1]), max(bounds[i, 0], bounds[i, 1]) + 1):
            color += img[bounds[i, 2], x, 0]
            pixels += 1

    return color // pixels


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
