import math

import numpy as np

from .shape import Shape
from numba import njit


class Ellipse(Shape):

    def __init__(self, A, B, C, D, E, F, alpha):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.alpha = alpha

    @staticmethod
    def random(w, h, alpha):
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        a = np.random.randint(1, w)
        b = np.random.randint(1, h)
        rotation = np.random.uniform(0, math.pi)
        return Ellipse.from_params(
            **{'cx': cx, 'cy': cy, 'a': a, 'b': b, 'rotation': rotation, 'alpha': alpha})

    @staticmethod
    def from_params(**params):
        # todo: maybe it would be better to move it to rasterize_ellipse?
        # todo: are the calculated coefficients needed elsewhere?
        b_b = params['b'] * params['b']
        a_a = params['a'] * params['a']
        sin2 = math.sin(2 * params['rotation'])
        h = params['cx']
        k = params['cy']
        cos_cos = math.cos(params['rotation']) * math.cos(params['rotation'])
        sin_sin = math.sin(params['rotation']) * math.sin(params['rotation'])
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
        return Ellipse(A, B, C, D, E, F, params['alpha'])

    def get_bounds(self):
        return rasterize_ellipse(self.A, self.B, self.C, self.D, self.E, self.F)

    def get_alpha(self):
        return self.alpha


@njit("i8[:,:](f8, f8, f8, f8, f8, f8)")
def rasterize_ellipse(A, B, C, D, E, F):
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
