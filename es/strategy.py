import logging
from abc import abstractmethod

import numpy as np

from shapes.shape import Curve
from shapes.shape import Ellipse
from shapes.shape import QuadrangleBrush, Quadrangle
from shapes.shape import Rectangle
from shapes.shape import Triangle
from shapes.util import stardardize

log = logging.getLogger(__name__)


class Strategy(object):

    def __init__(self, w, h, alpha, shape_mode, rng):
        self.w = w
        self.h = h
        self.alpha = alpha
        self.shape_mode = shape_mode
        self.rng = rng

    @abstractmethod
    def ask(self, scale):
        raise NotImplementedError

    @abstractmethod
    def tell(self, scores):
        raise NotImplementedError

    @abstractmethod
    def result(self):
        raise NotImplementedError


class RandomStrategy(Strategy):

    def __init__(self, n, w, h, alpha, rng, shape_mode, decay):
        super().__init__(w, h, alpha, shape_mode, rng)
        self.n = n
        self.shapes = None
        self.scores = None
        self.decay = decay
        self.scale = 1

    def ask(self, action):
        self.scale *= 1 / (1 + self.decay * action)
        log.debug(f'Scale of random shapes = {self.scale}')
        self.shapes = [self._random_shape() for _ in range(self.n)]
        return self.shapes

    def tell(self, scores):
        self.scores = scores

    def result(self):
        best = np.argmax(self.scores)
        return self.shapes[best], self.scores[best]

    def _random_shape(self):
        shape = self.rng.randint(6) if self.shape_mode == 0 else self.shape_mode - 1
        return {
            0: Triangle.random,
            1: Rectangle.random,
            2: Ellipse.random,
            3: Quadrangle.random,
            4: QuadrangleBrush.random,
            5: Curve.random,  # todo: choose only if resize == output size ?
        }[shape](w=self.w, h=self.h, alpha=self.alpha, rng=self.rng, scale=self.scale)


class SimpleEvolutionStrategy(Strategy):

    def __init__(self, initial_shape, w, h, alpha, n, sigma_factor, rng, shape_mode=0):
        super().__init__(w, h, alpha, shape_mode, rng)
        self.n = n
        self.shape = initial_shape.__class__
        self.mean = np.array(initial_shape.params(), dtype=np.float64)
        self.sigma = sigma_factor * self.shape.params_intervals()(w=self.w, h=self.h)
        self.shapes = None
        self.scores = None
        self.best = None

    def ask(self, scale=None):
        shapes = []
        for _ in range(self.n):
            params = [self.rng.normal(loc=mean, scale=sigma) for mean, sigma in
                      zip(self.mean, self.sigma)]
            shapes.append(self.shape.from_params(*params, self.alpha))
        return self.shapes

    def tell(self, scores):
        self.scores = scores
        self.best = np.armax(self.scores)
        self.mean = self.shapes[self.best].params()

    def result(self):
        return self.shapes[self.best], self.scores[self.best]


class EvolutionStrategy(Strategy):

    def __init__(self, initial_shape, w, h, alpha, n, sigma_factor, optimizer, rng, shape_mode=0):
        super().__init__(w, h, alpha, shape_mode, rng)
        self.n = n

        intervals = initial_shape.params_intervals()(w=self.w, h=self.h)
        self.sigma = sigma_factor * intervals
        self.shape = initial_shape.__class__
        self.optimizer = optimizer

        self.shapes = None
        self.scores = None
        self.eps = None
        self.best = None

    def ask(self, scale=None):
        self.eps = self.rng.normal(loc=0, scale=1, size=(self.n, len(self.optimizer.get_params())))
        shapes = []
        for i in range(self.n):
            params = [theta + sigma * eps for theta, sigma, eps in
                      zip(self.optimizer.get_params(), self.sigma, self.eps[i])]
            shape = self.shape.from_params(*params, self.alpha)
            shapes.append(shape)
        self.shapes = shapes
        return self.shapes

    def tell(self, scores):
        self.scores = scores
        standardized_scores = stardardize(scores)
        self.best = np.argmax(self.scores)
        gradient = np.dot(standardized_scores.T, self.eps) / (self.n * self.sigma)
        # gradient ascent
        self.optimizer.step(-gradient)

    def result(self):
        return self.shapes[self.best], self.scores[self.best]
