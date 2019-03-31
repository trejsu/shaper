from abc import abstractmethod

import numpy as np

from shapes.util import mse_full, l1_full, update_l1, update_mse


class Reward(object):

    @abstractmethod
    def init(self, canvas):
        raise NotImplementedError

    @abstractmethod
    def get(self, canvas, bounds):
        raise NotImplementedError

    @abstractmethod
    def undo(self):
        raise NotImplementedError


class DistanceReward(Reward):

    def __init__(self):
        self.distance = None
        self.prev_distance = None

    @property
    @abstractmethod
    def distance_fun(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def update_distance_fun(self):
        raise NotImplementedError

    @abstractmethod
    def reward(self):
        raise NotImplementedError

    def init(self, canvas):
        self.distance = self.distance_fun(
            target=canvas.target,
            x=canvas.img
        )

    def get(self, canvas, bounds):
        self.prev_distance = self.distance.copy()
        self.update_distance_fun(
            distance=self.distance,
            bounds=bounds,
            img=canvas.img,
            target=canvas.target
        )
        return self.reward()

    def undo(self):
        self.distance = self.prev_distance


class L2(DistanceReward):

    @property
    def distance_fun(self):
        return mse_full

    @property
    def update_distance_fun(self):
        return update_mse

    def reward(self):
        return -np.sqrt(self.distance.sum())


class L1(DistanceReward):

    @property
    def distance_fun(self):
        return l1_full

    @property
    def update_distance_fun(self):
        return update_l1

    def reward(self):
        return -np.average(self.distance)


class MSE(DistanceReward):

    @property
    def distance_fun(self):
        return mse_full

    @property
    def update_distance_fun(self):
        return update_mse

    def reward(self):
        return -np.average(self.distance)


class NN(Reward):
    def get(self, canvas):
        pass
