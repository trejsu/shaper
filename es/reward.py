from abc import abstractmethod

import numpy as np

from shapes.util import mse_full, l1_full, update_l1, update_mse


class Reward(object):

    @abstractmethod
    def get(self, bounds):
        raise NotImplementedError

    @abstractmethod
    def undo(self):
        raise NotImplementedError


class DistanceReward(Reward):

    def __init__(self, canvas):
        self.canvas = canvas
        self.distance = None
        self.prev_distance = None
        self.initialized = False

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

    def init(self):
        self.distance = self.distance_fun(
            target=self.canvas.target,
            x=self.canvas.img
        )
        self.initialized = True

    def get(self, bounds):
        if not self.initialized:
            self.init()
        self.prev_distance = self.distance.copy()
        self.update_distance_fun(
            distance=self.distance,
            bounds=bounds,
            img=self.canvas.img,
            target=self.canvas.target
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
