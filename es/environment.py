import logging
from abc import abstractmethod

import numpy as np

from es.reward import DistanceReward

log = logging.getLogger(__name__)


class Environment(object):
    def __init__(self, canvas, save_actions, num_shapes, reward_config):
        self.canvas = canvas
        self.save_actions = save_actions
        self.current_shape_num = 0
        self.shapes = [None] * num_shapes
        self.prev_img = None
        self.reward_config = reward_config

    def observation_shape(self):
        return self.canvas.size()

    @abstractmethod
    def evaluate(self, shape, n):
        score = self.step(shape, n)
        self._undo(n)
        return score

    @abstractmethod
    def evaluate_batch(self, shapes, n):
        current_reward = self.reward_config[n]
        if isinstance(current_reward, DistanceReward):
            return [self.evaluate(shape, n) for shape in shapes]
        else:
            X = np.empty((len(shapes), 28, 28, 1))

            for i, shape in enumerate(shapes):
                self.step(shape, n)
                x = self.canvas.img
                self._undo(n)
                X[i] = x[:, :, :1] / 255

            return current_reward.get(X)

    @abstractmethod
    def step(self, shape, n):
        self.prev_img = self.canvas.img.copy()
        bounds = self.canvas.add(shape)

        current_reward = self.reward_config[n]
        if isinstance(current_reward, DistanceReward):
            # todo: save actions wont be working when using model based rewards
            if self.save_actions:
                self._save_shape(shape)

            return current_reward.get(bounds)

    def save_in_size(self, output, size):
        if self.save_actions:
            log.debug(f'Saving image under {output} in size = {size}')

            canvas = self.canvas.clear_and_resize(size=size)
            w, h = canvas.size()

            for s in self.shapes:
                if s is None:
                    break
                else:
                    cls, normalized_params = s

                shape = cls.from_normalized_params(w, h, *normalized_params)
                canvas.add(shape=shape)

            canvas.save(output=output)
        else:
            raise Exception("Cannot save in size, save_actions set to False")

    @abstractmethod
    def _undo(self, n):
        self.canvas.img = self.prev_img
        current_reward = self.reward_config[n]
        current_reward.undo()
        if self.save_actions:
            self._remove_last_shape()

    def _remove_last_shape(self):
        self.current_shape_num -= 1

    def _save_shape(self, shape):
        self.shapes[self.current_shape_num] = (
            shape.__class__,
            shape.normalized_params(*self.observation_shape())
        )
        self.current_shape_num += 1
