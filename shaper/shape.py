from abc import abstractmethod

import numpy as np


class Shape(object):
    @staticmethod
    @abstractmethod
    def random(w, h):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_params(**params):
        raise NotImplementedError

    @abstractmethod
    def render(self, img):
        raise NotImplementedError


class Triangle(Shape):
    def __init__(self, points, color):
        self.points = points
        self.color = color

    def __str__(self):
        return f'Triangle: A = {self.points[0]}, B = {self.points[1]}, C = {self.points[2]}, ' \
               f'RGB = ({self.color[0]}, {self.color[1]}, {self.color[2]})'

    @staticmethod
    def random(w, h):
        xs = np.random.randint(w, size=(3, 1))
        ys = np.random.randint(h, size=(3, 1))
        points = np.concatenate((xs, ys), axis=1)
        assert points.shape == (3, 2), f'Shape of points: {points.shape}, expected: (3, 2)'
        color = np.random.randint(255, size=(3,))
        return Triangle(points, color)

    @staticmethod
    def from_params(**params):
        pass

    def render(self, img):
        def f(x1, y1, x2, y2, y):
            return ((y - y1) * x2 + (y2 - y) * x1) / (y2 - y1)

        changed = np.zeros(img.shape)

        upper = np.argmin(self.points[:, 1:])
        lower = np.argmax(self.points[:, 1:])
        third = [x for x in range(3) if x != upper and x != lower][0]

        upper_y = self.points[upper][1]
        upper_x = self.points[upper][0]
        lower_y = self.points[lower][1]
        lower_x = self.points[lower][0]

        for y in range(upper_y, lower_y):
            start_x = int(f(upper_x, upper_y, lower_x, lower_y, y))

            if y < self.points[third][1]:
                x1, y1 = self.points[upper]
                x2, y2 = self.points[third]
            else:
                x1, y1 = self.points[third]
                x2, y2 = self.points[lower]

            end_x = int(f(x1, y1, x2, y2, y))

            for x in range(start_x, end_x, 1 if start_x < end_x else -1):
                img[y, x] = self.color
                changed[y, x] = 1

        return changed
