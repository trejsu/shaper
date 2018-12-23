import numpy as np
from numba import njit

from .shape import Shape


# todo: add size
# todo: fix holes
class Curve(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha

    def __str__(self):
        return f'Curve: P0 = {self.points[0]}, P1 = {self.points[1]}, P2 = {self.points[2]}'

    @staticmethod
    def random(w, h, alpha):
        xs = np.random.randint(w, size=(3, 1))
        ys = np.random.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        return Curve(points, alpha)

    @staticmethod
    def from_params(*params):
        return Curve(points=np.array(params[:-1]).reshape(3, 2), alpha=params[-1])

    def get_bounds(self):
        return rasterize_curve(self.points, 0)

    def get_alpha(self):
        return self.alpha

    def args(self):
        return self.points.reshape(-1, ).astype(np.float64)

    @staticmethod
    def args_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h])


@njit("i8(f8, i8, i8, i8)")
def B(t, p0, p1, p2):
    if p0 == p1 == p2:
        return p0
    else:
        return int((1 - t) * (1 - t) * p0 + 2 * (1 - t) * t * p1 + t * t * p2)


@njit
def foo(y0, y1, y2):
    a = y0 - y1
    b = y0 - 2 * y1 + y2

    if a == 0:
        return y0
    if np.sign(a) != np.sign(b):
        return y0
    if b == 0:
        return y0
    if abs(a) > abs(b):
        return y1
    t = a / b
    return B(t, y0, y1, y2)


# @njit("i8[:,:](i8[:,:])")
# def rasterize_curve_2(points):
#     n = 20
#     ts = np.linspace(0, 1, n)
#     bounds = np.empty((n, 2), dtype=np.int64)
#
#     i = 0
#     for t in ts:
#         x = B(t, points[0, 0], points[1, 0], points[2, 0])
#         y = B(t, points[0, 1], points[1, 1], points[2, 1])


@njit("i8[:,:](i8[:,:], i8)")
def rasterize_curve(points, size):
    third = foo(points[0, 1], points[1, 1], points[2, 1])
    # print('y0', points[0,1])
    # print('y2', points[2,1])
    # print('third', third)

    num_t = 2 * max(abs(points[0, 0] - points[1, 0]), abs(points[1, 0] - points[2, 0]),
                    abs(points[0, 1] - points[1, 1]), abs(points[1, 1] - points[2, 1])) + 1
    ts = np.linspace(0, 1, num_t)

    num_bounds = abs(points[0, 1] - third) + abs(points[2, 1] - third) + 1

    # print('num_bounds', num_bounds)
    bounds = np.empty((num_bounds, 3), dtype=np.int64)

    i = 0
    prev_y = -9999999999999999
    for t in ts:

        x = B(t, points[0, 0], points[1, 0], points[2, 0])
        y = B(t, points[0, 1], points[1, 1], points[2, 1])

        # print('prev', prev_y)
        # print('y', y)
        # print('i', i)

        if y == prev_y:
            bounds[i - 1, 0] = min(bounds[i - 1, 0], x - size)
            bounds[i - 1, 1] = max(bounds[i - 1, 1], x + size)
        else:
            if i >= len(bounds):
                print('points', points)
                print('third', third)
                print('num_bounds', num_bounds)
                print('prev', prev_y)
                print('y', y)
                print('i', i)
                print('bounds', bounds)
                raise Exception

            bounds[i, 0] = x - size
            bounds[i, 1] = x + size
            bounds[i, 2] = y
            i += 1

        prev_y = y

    return bounds


class TripleCurve(Shape):

    def __init__(self, points, alpha):
        self.points = points.astype(np.int64)
        self.alpha = alpha
        self.size = np.random.randint(8)

    @staticmethod
    def random(w, h, alpha):
        xs = np.random.randint(w, size=(3, 1))
        ys = np.random.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        # size = np.random.randint(min(w, h) // 2)
        return TripleCurve(points, alpha)

    @staticmethod
    def from_params(*params):
        return TripleCurve(points=np.array(params[:-1]).reshape(3, 2), alpha=params[-1])

    def get_bounds(self):
        bounds = rasterize_curve(self.points, self.size)
        for i in range(-self.size, self.size + 1):
            if i == 0:
                continue
            else:
                new_points = np.array([[x, y + i] for x, y in self.points], dtype=np.int64)
                new_bounds = rasterize_curve(new_points, self.size)
                bounds = np.concatenate((bounds, new_bounds), axis=0)
        return bounds

    def get_alpha(self):
        return self.alpha

    def args(self):
        reshape = self.points.astype(np.float64).reshape(-1, )
        print(f'reshaped dtype {reshape.dtype}')
        return reshape

    @staticmethod
    def args_intervals():
        return lambda w, h: np.array([w, h, w, h, w, h])
