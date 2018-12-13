import math
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

    def render(self, img, target):
        bounds = rasterize_rectangle(self.points)
        crop_bounds(bounds=bounds, h=img.shape[0], w=img.shape[1])
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


@njit("i8[:,:](i8[:,:])")
def rasterize_rectangle(points):
    upper = np.argmin(points[:, 1])
    lower = np.argmax(points[:, 1])
    left = np.argmin(points[:, 0])
    right = 6 - (upper + lower + left)

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


@njit("(i8[:,:], i8, i8)")
def crop_bounds(bounds, h, w):
    for i in range(len(bounds)):
        bounds[i, 0] = max(0, min(w - 1, bounds[i, 0]))
        bounds[i, 1] = max(0, min(w - 1, bounds[i, 1]))
        bounds[i, 2] = max(0, min(h - 1, bounds[i, 2]))
