from abc import abstractmethod

import numpy as np
from numba import njit

from shaper.util import bounds_to_pixels, MIN_VALUE, timeit
from .curve import Curve
from .ellipse import Ellipse
from .shape import Shape, crop_bounds


class Brush(Shape):

    def __init__(self, path, size, alpha):
        self.path = path
        self.size = max(1, int(size))
        self.alpha = alpha

    @classmethod
    @timeit
    def random(cls, w, h, alpha):
        path = Curve.random(w=w, h=h, alpha=alpha)
        size = np.random.randint(1, min(w, h) // 2)
        return cls(path=path, size=size, alpha=alpha)

    @classmethod
    @timeit
    def from_params(cls, *params):
        return cls(path=Curve.from_params(*params[:-2], params[-1]), size=params[-2],
                   alpha=params[-1])

    @classmethod
    @timeit
    def from_normalized_params(cls, w, h, *params):
        return cls(
            path=Curve.from_normalized_params(w, h, *params[:-2], params[-1]),
            size=int(params[-2] * (min(w, h) // 2)),
            alpha=params[-1]
        )

    @timeit
    def get_bounds(self, h, w):
        centers = self.get_shapes_centers(h, w)
        shapes = [self.get_shape(x=x, y=y, size=self.size) for x, y in centers]
        num_bounds = (4 * self.size) * len(shapes)

        bounds = np.empty(shape=(num_bounds, 3), dtype=np.int64)

        i = 0
        for shape in shapes:
            b = shape.get_bounds()
            b_len = b.shape[0]
            bounds[i:i + b_len] = b
            i += b_len

        bounds.resize((i, 3))

        return merge_bounds_for_simple_path(bounds=bounds)

    @timeit
    def get_shapes_centers(self, h, w):
        path_bounds = self.path.get_bounds()
        crop_bounds(bounds=path_bounds, h=h, w=w)
        path_pixels = bounds_to_pixels(path_bounds)
        shapes_to_skip = max(1, self.size // 2)
        return path_pixels[::shapes_to_skip]

    def get_alpha(self):
        return self.alpha

    @timeit
    def params(self):
        return np.append(self.path.params(), self.size)

    @timeit
    def normalized_params(self, w, h):
        return np.append(self.path.normalized_params(w, h)[:-1],
                         [self.size / (min(w, h) // 2), self.alpha])

    @staticmethod
    @timeit
    def params_intervals():
        return lambda w, h: np.append(Curve.params_intervals()(w, h), min(w, h) // 2)

    @abstractmethod
    @timeit
    def get_shape(self, x, y, size):
        raise NotImplementedError

    def __str__(self):
        return f'Brush(path={self.path}, size={self.size})'


class EllipseBrush(Brush):
    brush_type = Ellipse

    def __init__(self, path, size, alpha):
        super().__init__(path=path, size=size, alpha=alpha)

    @timeit
    def get_shape(self, x, y, size):
        return self.brush_type(a=size, b=size, h=x, k=y, r=0, alpha=self.alpha)


@njit("i8[:,:](i8[:,:])")
def merge_bounds_for_simple_path(bounds):
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
            merged[j, 0] = start
            merged[j, 1] = end
        else:
            merged[j, 0] = min(start, end, current[0])
            merged[j, 1] = max(start, end, current[1])

    return merged


# todo: fix/finish/reimplement
@njit("i8[:,:](i8[:,:], i8, i8, i8)")
def merge_bounds_for_curved_path(bounds, path_extremum, path_num_bounds, size):
    min_y = np.min(bounds[:, 2])
    max_y = np.max(bounds[:, 2])

    num_y = path_num_bounds + 4 * size

    ext_plus_size = path_extremum + size
    ext_minus_size = path_extremum - size
    if ext_plus_size == max_y or ext_plus_size == max_y + 1 or ext_plus_size == max_y - 1:
        extremum = max_y
    elif ext_minus_size == min_y or ext_minus_size == min_y + 1 or ext_minus_size == min_y - 1:
        extremum = min_y
    else:
        print('path_extremum, max_y, min_y', path_extremum, max_y, min_y)
        raise Exception

    merged = np.full(shape=(num_y, 3), fill_value=MIN_VALUE, dtype=np.int64)

    longer_part_len = max_y - min_y
    shorter_part_len = num_y - longer_part_len + 1
    merged[0:longer_part_len, 2] = np.arange(min_y, max_y)
    y_start = extremum
    y_end = extremum - shorter_part_len if extremum == max_y else extremum + shorter_part_len
    merged[longer_part_len:num_y + 1, 2] = np.arange(min(y_start, y_end), max(y_start, y_end) + 1)

    for i in range(len(bounds)):
        start, end, y = bounds[i]
        j = y - min_y
        current = merged[j]
        if current[0] == MIN_VALUE:
            merged[j, 0] = start
            merged[j, 1] = end
        else:
            merged[j, 0] = min(start, end, current[0])
            merged[j, 1] = max(start, end, current[1])

    return merged
