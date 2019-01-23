from abc import abstractmethod

import numpy as np
from .util import timeit


class Optimizer(object):

    def __init__(self, initial_params):
        self._params = initial_params.copy()

    def get_params(self):
        return self._params.copy()

    @abstractmethod
    def step(self, gradients):
        raise NotImplementedError()


class GradientDescent(Optimizer):

    def __init__(self, initial_params, learning_rate):
        super().__init__(initial_params)
        self.learning_rate = learning_rate

    def step(self, gradients):
        self._params -= self.learning_rate * gradients


class Momentum(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma=.9):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.v = np.zeros(initial_params.shape)

    def step(self, gradients):
        self.v = self.gamma * self.v + self.learning_rate * gradients
        self._params -= self.v


class Nesterov(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma=.9):
        super().__init__(initial_params)
        self.training_phase = True
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.v = np.zeros(initial_params.shape)

    def step(self, gradients):
        self.v = self.gamma * self.v + self.learning_rate * gradients
        self._params -= self.v

    def _get_training_params(self):
        return self._params - self.gamma * self.v

    def _get_test_params(self):
        return self._params

    def get_params(self):
        if self.training_phase:
            return self._get_training_params()
        else:
            return self._get_test_params()


class Adagrad(Optimizer):

    def __init__(self, initial_params, learning_rate, epsilon=1e-8):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.h = np.zeros(initial_params.shape)

    def step(self, gradients):
        self.h += np.square(gradients)
        self._params -= self.learning_rate * gradients / np.sqrt(self.h + self.epsilon)


class RMSProp(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma=.9, epsilon=1e-8):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.h = np.zeros(initial_params.shape)

    def step(self, gradients):
        self.h = self.gamma * self.h + (1 - self.gamma) * np.square(gradients)
        self._params -= self.learning_rate * gradients / np.sqrt(self.h + self.epsilon)


class Adadelta(Optimizer):

    def __init__(self, initial_params, learning_rate, gamma=.95, epsilon=1e-4):
        super().__init__(initial_params)
        self.gamma = gamma
        self.epsilon = epsilon
        self.h = np.zeros(initial_params.shape)
        self.d = np.zeros(initial_params.shape)

    def step(self, gradients):
        self.h = self.gamma * self.h + (1 - self.gamma) * np.square(gradients)
        old_params = self._params
        self._params = old_params - np.sqrt(self.d + self.epsilon) * gradients / np.sqrt(
            self.h + self.epsilon)
        self.d = self.gamma * self.d + (1 - self.gamma) * np.square(self._params - old_params)


class Adam(Optimizer):

    def __init__(self, initial_params, learning_rate, beta1=.9, beta2=.999, epsilon=1e-8):
        super().__init__(initial_params)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(initial_params.shape)
        self.v = np.zeros(initial_params.shape)
        self.num_steps = 0

    @timeit
    def step(self, gradients):
        self.num_steps += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradients)
        m = self.m / (1 - self.beta1 ** self.num_steps)
        v = self.v / (1 - self.beta2 ** self.num_steps)
        self._params -= self.learning_rate * m / (np.sqrt(np.array(v)) + self.epsilon)
