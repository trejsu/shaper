from abc import abstractmethod

from numba import njit


class Shape(object):
    @staticmethod
    @abstractmethod
    def random(w, h, alpha):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_params(*params):
        raise NotImplementedError

    def render(self, img, target):
        bounds = self.get_bounds(h=img.shape[0], w=img.shape[1])
        crop_bounds(bounds=bounds, h=img.shape[0], w=img.shape[1])
        color = average_color(img=target, bounds=bounds)
        alpha = self.get_alpha()
        assert 0 <= alpha <= 1, f'alpha out of bounds = {alpha}'
        render(img=img, bounds=bounds, color=color, alpha=alpha)
        return bounds

    @abstractmethod
    def get_bounds(self, h, w):
        raise NotImplementedError

    @abstractmethod
    def get_alpha(self):
        raise NotImplementedError

    @abstractmethod
    def args(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def args_intervals():
        raise NotImplementedError


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

    color = r // pixels, g // pixels, b // pixels
    return color


@njit("(i8[:,:], i8, i8)")
def crop_bounds(bounds, h, w):
    for i in range(len(bounds)):
        bounds[i, 0] = max(0, min(w - 1, bounds[i, 0]))
        bounds[i, 1] = max(0, min(w - 1, bounds[i, 1]))
        bounds[i, 2] = max(0, min(h - 1, bounds[i, 2]))
