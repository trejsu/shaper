import logging
from abc import abstractmethod

import numpy as np

from shaper.shape.curve import Curve
from shaper.shape.ellipse import Ellipse
from shaper.shape.rectangle import Rectangle
from shaper.shape.triangle import Triangle

log = logging.getLogger(__name__)


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
            2: Ellipse.random,
            3: Curve.random
        }[np.random.randint(4)](w=self.w, h=self.h, alpha=self.alpha)

    @staticmethod
    def _shape_class(shape):
        classes = [Triangle, Rectangle, Ellipse, Curve]
        for cls in classes:
            if isinstance(shape, cls):
                return cls


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

    def __init__(self, initial_shape, w, h, alpha, n, sigma_factor=.03):
        self.n = n
        self.w = w
        self.h = h
        self.alpha = alpha
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

    def __init__(self, initial_shape, w, h, alpha, n, lr, sigma_factor):
        self.w = w
        self.h = h
        self.alpha = alpha
        self.n = n
        self.lr = lr

        self.theta = np.array(initial_shape.args(), dtype=np.float64)
        self.sigma = sigma_factor * initial_shape.args_intervals()(w=self.w, h=self.h)
        self.shape = Strategy._shape_class(initial_shape)

        self.shapes = None
        self.scores = None
        self.eps = None
        self.best = None

    def ask(self):
        self.eps = np.random.normal(loc=0, scale=1, size=(self.n, len(self.theta)))
        shapes = []
        for i in range(self.n):
            args = [theta + sigma * eps for theta, sigma, eps in
                    zip(self.theta, self.sigma, self.eps[i])]
            shape = self.shape.from_params(*args, self.alpha)
            shapes.append(shape)
        self.shapes = shapes
        return self.shapes

    def tell(self, scores):
        self.scores = scores
        normalized_scores = -((np.array(scores) - np.mean(scores)) / np.std(scores))
        self.best = np.argmin(self.scores)
        update = self.lr / (self.n * self.sigma) * np.dot(normalized_scores.T, self.eps)
        self.theta += update

    def result(self):
        return self.shapes[self.best], self.scores[self.best]
