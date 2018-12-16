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

    def _random_shape(self):
        return {
            0: Triangle.random,
            1: Rectangle.random,
            2: Ellipse.random
        }[np.random.randint(3)](w=self.w, h=self.h, alpha=self.alpha)
