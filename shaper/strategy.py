from abc import abstractmethod

import numpy as np

from shaper.shape.curve import Curve
from shaper.shape.ellipse import Ellipse
from shaper.shape.rectangle import Rectangle
from shaper.shape.triangle import Triangle


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


# todo: find why it stops
# todo: find good hyperparameters
class EvolutionStrategy(Strategy):

    def __init__(self, initial_shape, w, h, alpha, n=100, lr=0.1, noise_sigma=0.1,
        shape_sigma_factor=0.03):
        self.w = w
        self.h = h
        self.alpha = alpha
        self.n = n
        self.lr = lr
        self.noise_sigma = noise_sigma
        self.mean = np.array(initial_shape.args(), dtype=np.float64).reshape(1, -1)
        self.shape_sigma = shape_sigma_factor * initial_shape.args_intervals()(w=self.w, h=self.h)
        self.shape = Strategy._shape_class(initial_shape)
        self.shapes = None
        self.scores = None
        self.eps = None
        self.best = None

    def ask(self):
        self.eps = np.random.normal(loc=0, scale=1, size=(self.n, 1))
        shapes = []
        for _ in range(self.n):
            args = [np.random.normal(loc=mean + self.noise_sigma * eps, scale=sigma) for
                    mean, sigma, eps in zip(self.mean[0], self.shape_sigma, self.eps[:, 0])]
            shapes.append(self.shape.from_params(*args, self.alpha))
        self.shapes = shapes
        return self.shapes

    def tell(self, scores):
        self.scores = scores
        self.best = np.argmin(self.scores)
        self.mean += self.lr / (self.n * self.noise_sigma) * np.dot(
            np.array(self.scores).reshape(1, -1), self.eps)

    def result(self):
        return self.shapes[self.best], self.scores[self.best]
