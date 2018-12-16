import math

import numpy as np
from numba import njit

from .shape import Shape, f


class Rectangle(Shape):

    def __init__(self, points, alpha):
        self.points = points
        self.alpha = alpha

    def __str__(self):
        return f'Triangle: A = {self.points[0]}, B = {self.points[1]}, ' \
               f'C = {self.points[2]}, D = {self.points[3]}'

    @staticmethod
    def random(w, h, alpha):
        def deg_to_rad(deg):
            return deg * math.pi / 180

        def rad_to_deg(rad):
            return rad * 180 / math.pi

        cx = np.random.randint(w)
        cy = np.random.randint(h)
        rw = np.random.randint(1, w)
        rh = np.random.randint(1, h)
        rot = np.random.uniform(0, math.pi)

        # todo: maybe it would be better to move it to rasterize_rectangle?
        # todo: are the calculated points needed elsewhere?
        x1 = cx + math.sin(
            (rot - deg_to_rad(90)) + math.asin(rh / (math.sqrt(rh * rh + rw * rw)))) * (
                 math.sqrt(rh * rh + rw * rw) / 2)
        y1 = cy - math.cos(
            (rot - deg_to_rad(90)) + math.asin(rh / (math.sqrt(rh * rh + rw * rw)))) * (
                 math.sqrt(rh * rh + rw * rw) / 2)
        x2 = x1 + math.cos(rot) * rw
        y2 = y1 + math.sin(rot) * rw
        x3 = x2 + math.cos(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        y3 = y2 + math.sin(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        x4 = x1 + math.cos(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        y4 = y1 + math.sin(deg_to_rad(rad_to_deg(rot) + 90)) * rh
        return Rectangle(points=np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int64),
                         alpha=alpha)

    @staticmethod
    def from_params(**params):
        pass

    def get_bounds(self):
        return rasterize_rectangle(self.points)

    def get_alpha(self):
        return self.alpha


@njit("i8[:,:](i8[:,:])")
def rasterize_rectangle(points):
    upper = np.argmin(points[:, 1])
    lower = np.argmax(points[:, 1])
    left = np.argmin(points[:, 0])
    right = 6 - (upper + lower + left)

    # todo: add +1 to bounds size
    bounds = np.empty((points[lower][1] - points[upper][1], 3), dtype=np.int64)

    i = 0
    for y in range(points[upper][1], points[lower][1]):

        if y < points[left][1]:
            x1, y1 = points[upper]
            x2, y2 = points[left]
        else:
            x1, y1 = points[left]
            x2, y2 = points[lower]

        start_x = f(x1, y1, x2, y2, y)

        if y < points[right][1]:
            x1, y1 = points[upper]
            x2, y2 = points[right]
        else:
            x1, y1 = points[right]
            x2, y2 = points[lower]

        end_x = f(x1, y1, x2, y2, y)

        bounds[i, 0] = start_x
        bounds[i, 1] = end_x
        bounds[i, 2] = y
        i += 1

    return bounds
