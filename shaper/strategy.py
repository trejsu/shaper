import logging
from abc import abstractmethod

import numpy as np

from shaper.shape.brush import EllipseBrush
from shaper.shape.curve import Curve
from shaper.shape.ellipse import Ellipse
from shaper.shape.rectangle import Rectangle
from shaper.shape.triangle import Triangle
from shaper.util import normalize

log = logging.getLogger(__name__)


class Strategy(object):

    def __init__(self, w, h, alpha, shape_mode):
        self.w = w
        self.h = h
        self.alpha = alpha
        self.shape_mode = shape_mode

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
        shape = np.random.randint(5) if self.shape_mode == 0 else self.shape_mode - 1
        return {
            0: Triangle.random,
            1: Rectangle.random,
            2: Ellipse.random,
            3: Curve.random,
            4: EllipseBrush.random
        }[shape](w=self.w, h=self.h, alpha=self.alpha)

    @staticmethod
    def _shape_class(shape):
        classes = [Triangle, Rectangle, Ellipse, Curve, EllipseBrush]
        for cls in classes:
            if isinstance(shape, cls):
                return cls


class RandomStrategy(Strategy):

    def __init__(self, num_shapes, w, h, alpha, shape_mode=0):
        super().__init__(w, h, alpha, shape_mode)
        self.num_shapes = num_shapes
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

    def __init__(self, initial_shape, w, h, alpha, n, sigma_factor, shape_mode=0):
        super().__init__(w, h, alpha, shape_mode)
        self.n = n
        self.shape = Strategy._shape_class(initial_shape)
        self.mean = np.array(initial_shape.args(), dtype=np.float64)
        self.sigma = sigma_factor * self.shape.args_intervals()(w=self.w, h=self.h)
        self.shapes = None
        self.scores = None
        self.best = None

    def ask(self):
        shapes = []
        for _ in range(self.n):
            args = [np.random.normal(loc=mean, scale=sigma) for mean, sigma in
                    zip(self.mean, self.sigma)]
            shapes.append(self.shape.from_params(*args, self.alpha))
        self.shapes = shapes
        return self.shapes

    def tell(self, scores):
        self.scores = scores
        self.best = np.argmin(self.scores)
        self.mean = self.shapes[self.best].args()

    def result(self):
        return self.shapes[self.best], self.scores[self.best]


class EvolutionStrategy(Strategy):

    def __init__(self, initial_shape, w, h, alpha, n, sigma_factor, optimizer, shape_mode=0):
        super().__init__(w, h, alpha, shape_mode)
        self.n = n

        self.sigma = sigma_factor * initial_shape.args_intervals()(w=self.w, h=self.h)
        self.shape = Strategy._shape_class(initial_shape)
        self.optimizer = optimizer

        self.shapes = None
        self.scores = None
        self.eps = None
        self.best = None

    def ask(self):
        self.eps = np.random.normal(loc=0, scale=1, size=(self.n, len(self.optimizer.get_params())))
        shapes = []
        for i in range(self.n):
            args = [theta + sigma * eps for theta, sigma, eps in
                    zip(self.optimizer.get_params(), self.sigma, self.eps[i])]
            shape = self.shape.from_params(*args, self.alpha)
            shapes.append(shape)
        self.shapes = shapes
        return self.shapes

    def tell(self, scores):
        self.scores = scores
        normalized_scores = normalize(scores)
        self.best = np.argmin(self.scores)
        gradient = np.dot(normalized_scores.T, self.eps) / (self.n * self.sigma)
        self.optimizer.step(gradient)

    def result(self):
        return self.shapes[self.best], self.scores[self.best]
