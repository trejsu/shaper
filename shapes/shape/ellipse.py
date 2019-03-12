import math

import numpy as np
from numba import njit

from .shape import Shape


class Ellipse(Shape):

    def __init__(self, a, b, h, k, r, alpha):
        self.a = max(1, int(a))
        self.b = max(1, int(b))
        self.h = int(h)
        self.k = int(k)
        self.r = r
        self.alpha = alpha

    @staticmethod
    def random(w, h, alpha, rng, scale=1):
        center_x = rng.randint(w)
        center_y = rng.randint(h)
        a = scale * rng.randint(1, w)
        b = scale * rng.randint(1, h)
        rotation = rng.uniform(0, math.pi)
        return Ellipse(a=a, b=b, h=center_x, k=center_y, r=rotation, alpha=alpha)

    @staticmethod
    def from_params(*params):
        return Ellipse(*params)

    @staticmethod
    def from_normalized_params(w, h, *params):
        return Ellipse(
            a=int(params[0] * w),
            b=int(params[1] * h),
            h=int(params[2] * w),
            k=int(params[3] * h),
            r=params[4] * math.pi,
            alpha=params[5]
        )

    def get_bounds(self, h=None, w=None):
        return rasterize_ellipse(a=self.a, b=self.b, h=self.h, k=self.k, r=self.r)

    def get_alpha(self):
        return self.alpha

    def params(self):
        return np.array([self.a, self.b, self.h, self.k, self.r], dtype=np.float64)

    def normalized_params(self, w, h):
        return np.append(np.array(
            [self.a, self.b, self.h, self.k, self.r],
            dtype=np.float64
        ) / np.array([w, h, w, h, math.pi]), self.alpha)

    @staticmethod
    def params_intervals():
        return lambda w, h: np.array([w, h, w - 1, h - 1, math.pi])

    def __str__(self):
        return f'Ellipse(a={self.a}, b={self.b}, cx={self.h}, cy={self.k}, rotation={self.r})'


@njit("i8[:,:](f8, f8, f8, f8, f8)")
def rasterize_ellipse(a, b, h, k, r):
    b_b = b * b
    a_a = a * a
    sin2 = math.sin(2 * r)
    cos_cos = math.cos(r) * math.cos(r)
    sin_sin = math.sin(r) * math.sin(r)
    h_h = h * h
    k_k = k * k
    h_k_sin2 = h * k * sin2

    A = b_b * cos_cos + a_a * sin_sin
    B = a_a * cos_cos + b_b * sin_sin
    C = -2 * b_b * h * cos_cos - (a_a - b_b) * k * sin2 - 2 * a_a * h * sin_sin
    D = -2 * a_a * k * cos_cos - (a_a - b_b) * h * sin2 - 2 * b_b * k * sin_sin
    E = (a_a - b_b) * sin2
    F = (b_b * h_h + a_a * k_k) * cos_cos - b_b * h_k_sin2 + (
        a_a * h_h + b_b * k_k) * sin_sin + a_a * (-b_b + h_k_sin2)

    a_y = 4 * A * B - E * E
    b_y = 4 * A * D - 2 * C * E
    c_y = 4 * A * F - C * C
    y_roots = np.roots(np.array([a_y, b_y, c_y]))

    lower_y = int(np.max(y_roots))
    upper_y = int(np.min(y_roots))

    bounds = np.empty((lower_y - upper_y + 1, 3), dtype=np.int64)

    i = 0
    for y in range(upper_y, lower_y + 1):
        a_x = A
        b_x = C + E * y
        c_x = B * y * y + D * y + F

        x_roots = np.real(
            np.roots(np.array([np.complex(a_x, 0), np.complex(b_x, 0), np.complex(c_x, 0)])))

        bounds[i, 0] = int(x_roots[1])
        bounds[i, 1] = int(x_roots[0])
        bounds[i, 2] = y
        i += 1

    return bounds