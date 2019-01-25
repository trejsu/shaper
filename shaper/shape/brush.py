import logging

import numpy as np
from numba import njit

from shaper.util import MIN_VALUE, timeit
from .curve import Curve
from .quadrangle import Quadrangle
from .shape import Shape

log = logging.getLogger(__name__)


class Brush(Shape):
    SIZE_SCALE = 4

    def __init__(self, path, size, alpha):
        self.path = path
        self.size = max(1, int(size))
        self.alpha = alpha

    @classmethod
    @timeit
    def random(cls, w, h, alpha, rng, scale=1):
        path = Curve.random(w=w, h=h, alpha=alpha, rng=rng, scale=scale)
        size = rng.randint(1, min(w, h) // Brush.SIZE_SCALE) * scale
        return cls(path=path, size=size, alpha=alpha)

    @classmethod
    @timeit
    def from_params(cls, *params):
        return cls(
            path=Curve.from_params(*params[:-2], params[-1]),
            size=params[-2],
            alpha=params[-1]
        )

    @classmethod
    @timeit
    def from_normalized_params(cls, w, h, *params):
        return cls(
            path=Curve.from_normalized_params(w, h, *params[:-2], params[-1]),
            size=int(params[-2] * (min(w, h) // Brush.SIZE_SCALE)),
            alpha=params[-1]
        )

    @timeit
    def get_bounds(self, h, w):
        raise NotImplementedError

    def get_alpha(self):
        return self.alpha

    @timeit
    def params(self):
        return np.append(self.path.params(), self.size)

    @timeit
    def normalized_params(self, w, h):
        return np.append(self.path.normalized_params(w, h)[:-1],
                         [self.size / (min(w, h) // Brush.SIZE_SCALE), self.alpha])

    @staticmethod
    @timeit
    def params_intervals():
        return lambda w, h: np.append(Curve.params_intervals()(w, h), min(w, h) // Brush.SIZE_SCALE)

    def __str__(self):
        return f'Brush(path={self.path}, size={self.size})'


class QuadrangleBrush(Brush):

    def __init__(self, path, size, alpha):
        super().__init__(path=path, size=size, alpha=alpha)

    def get_bounds(self, h, w):
        num_path_pixels = 10
        path_pixels = self.path.get_points(n=num_path_pixels)

        if path_pixels.shape[0] < 3:
            log.debug('Path pixels < 3, rectangle brush empty.')
            return np.empty(shape=(0, 3), dtype=np.int64)
        points = generate_quadrangle_points(path_pixels, self.size)

        shapes = [Quadrangle.without_edge(
            points=p,
            edge=[0, 1],
            alpha=self.get_alpha
        ) for p in points]

        # shapes = [Quadrangle.from_params(p, self.get_alpha) for p in points]

        num_bounds = 12 * h * len(shapes)
        bounds = np.empty(shape=(num_bounds, 3), dtype=np.int64)

        i = 0
        for shape in shapes:
            b = shape.get_bounds()
            b_len = b.shape[0]
            bounds[i:i + b_len] = b
            i += b_len

        bounds.resize((i, 3))

        return bounds

    def get_shape(self, x, y, size):
        pass


@njit("f8[:,:](f8[:,:], f8[:,:], i8, i8)")
def find_orthogonal_vector(v, S, size, coordinate):
    temp = np.copy(v[:, 0])
    v[:, 0] = v[:, 1]
    v[:, 1] = temp
    v[:, coordinate] = -v[:, coordinate]
    norm = np.sqrt(np.square(v).sum(axis=1)).reshape(-1, 1)
    normalized = v if np.any(norm == 0) else v / norm
    return normalized * size + S


@njit("i8[:,:,:](i8[:,:], i8)")
def generate_quadrangle_points(path_points, size):
    A = path_points[1:]
    B = path_points[:-1]
    S = (A + B) / 2
    AB = B - S

    left = find_orthogonal_vector(v=AB.copy(), S=S, size=size, coordinate=0)
    right = find_orthogonal_vector(v=AB.copy(), S=S, size=size, coordinate=1)

    points = np.empty(shape=(path_points.shape[0] - 2, 4, 2), dtype=np.int64)

    for i in range(points.shape[0]):
        points[i, 0] = left[i]
        points[i, 1] = right[i]
        points[i, 2] = right[i + 1]
        points[i, 3] = left[i + 1]

    return points


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
