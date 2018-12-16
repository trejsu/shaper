from abc import abstractmethod

import numpy as np

from .ellipse import Ellipse
from .rectangle import Rectangle
from .triangle import Triangle


class Strategy(object):

    @abstractmethod
    def ask(self):
        raise NotImplementedError

    @abstractmethod
    def tell(self, scores):
        raise NotImplementedError

    @abstractmethod
    def result(self):
        raise NotImplementedError

    def _random_shape(self):
        return {
            0: Triangle.random,
            1: Rectangle.random,
            2: Ellipse.random
        }[np.random.randint(3)](w=self.w, h=self.h, alpha=self.alpha)


class RandomStrategy(Strategy):

    def __init__(self, num_shapes, w, h, alpha):
        self.num_shapes = num_shapes
        self.w = w
        self.h = h
        self.alpha = alpha
        self.shapes = None
        self.scores = None

    def ask(self):
        self.shapes = [self._random_shape() for _ in range(self.num_shapes)]
        return self.shapes

    def tell(self, scores):
        self.scores = scores

    def result(self):
        best = np.argmin(self.scores)
        return self.shapes[best], self.scores[best]


class SimpleEvolutionStrategy(Strategy):

    def __init__(self, num_shapes, w, h, alpha, var_factor=.03):
        self.num_shapes = num_shapes
        self.w = w
        self.h = h
        self.alpha = alpha
        self.shapes = None
        self.scores = None
        self.best = None
        self.means = None
        self.vars = None
        self.var_factor = var_factor
        self.shape = None

    def ask(self):
        if self.means is None:
            self.shapes = [self._random_shape() for _ in range(self.num_shapes)]
        else:
            shapes = []
            for _ in range(self.num_shapes):
                args = [np.random.normal(loc=mean, scale=var) for mean, var in
                        zip(self.means, self.vars)]
                shapes.append(self.shape.from_params(*args, self.alpha))
            self.shapes = shapes
        return self.shapes

    def tell(self, scores):
        self.scores = scores
        self.best = np.argmin(self.scores)
        self.means = self.shapes[self.best].args()
        if self.vars is None:
            self.shape = self._shape_class(self.shapes[self.best])
            self.vars = self.var_factor * self.shape.args_intervals()(w=self.w, h=self.h)

    def result(self):
        return self.shapes[self.best], self.scores[self.best]

    @staticmethod
    def _shape_class(shape):
        classes = [Triangle, Rectangle, Ellipse]
        for cls in classes:
            if isinstance(shape, cls):
                return cls
