import logging
from abc import abstractmethod

import numpy as np

from shapes.util import mse_full, l1_full, update_l1, update_mse

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
        raise NotImplementedError

    @abstractmethod
    def evaluate_batch(self, shapes, n):
        raise NotImplementedError

    @abstractmethod
    def step(self, shape, n):
        raise NotImplementedError

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
        raise NotImplementedError

    def _remove_last_shape(self):
        self.current_shape_num -= 1

    def _save_shape(self, shape):
        self.shapes[self.current_shape_num] = (
            shape.__class__,
            shape.normalized_params(*self.observation_shape())
        )
        self.current_shape_num += 1


# todo: remove after adding nn rewards
class DistanceEnv(Environment):

    def __init__(self, canvas, save_actions, num_shapes, reward_config):
        super().__init__(canvas, save_actions, num_shapes, reward_config)

    def evaluate(self, shape, n):
        score = self.step(shape, n)
        self._undo(n)
        return score

    def evaluate_batch(self, shapes, n):
        return [self.evaluate(shape, n) for shape in shapes]

    def step(self, shape, n):
        self.prev_img = self.canvas.img.copy()
        bounds = self.canvas.add(shape)

        if self.save_actions:
            self._save_shape(shape)

        current_reward = self.reward_config[n]
        return current_reward.get(bounds)

    def _undo(self, n):
        self.canvas.img = self.prev_img
        current_reward = self.reward_config[n]
        current_reward.undo()
        if self.save_actions:
            self._remove_last_shape()


class NNEnv(Environment):

    def __init__(self, canvas, save_actions, num_shapes, model_initializer):
        super().__init__(canvas, save_actions, num_shapes)
        self.init_model = model_initializer
        self.__model = None

    @property
    def model(self):
        if self.__model is None:
            self.__model = self.init_model()
        return self.__model

    def evaluate(self, shape):
        raise NotImplementedError

    def evaluate_batch(self, shapes):
        X = np.empty((len(shapes), 28, 28, 1))

        for i, shape in enumerate(shapes):
            self.step(shape)
            x = self.canvas.img
            self._undo()
            X[i] = x[:, :, :1] / 255

        # probability that given x is a specific number
        model_output = self.model.predict(X)
        return model_output

    def step(self, shape):
        self.prev_img = self.canvas.img.copy()
        self.canvas.add(shape)

    def _undo(self):
        self.canvas.img = self.prev_img
        if self.save_actions:
            self._remove_last_shape()

    def init(self):
        raise NotImplementedError


class MixedEnv(Environment):

    def __init__(self, canvas, save_actions, num_shapes, metric, model_initializer):
        super().__init__(canvas, save_actions, num_shapes)
        self.metric = metric
        self.distance = (l1_full if metric == 'l1' else mse_full)(
            target=self.canvas.target,
            x=self.canvas.img
        )
        self.update_distance = update_l1 if metric == 'l1' else update_mse
        self.prev_distance = None
        self.init_model = model_initializer
        self.__model = None

    @property
    def model(self):
        if self.__model is None:
            self.__model = self.init_model()
        return self.__model

    def init(self):
        raise NotImplementedError

    def evaluate(self, shape):
        raise NotImplementedError

    def evaluate_batch(self, shapes):
        X = np.empty((len(shapes), 28, 28, 1))
        distances = np.empty((len(shapes, )))

        for i, shape in enumerate(shapes):
            distances[i] = self.step(shape)
            x = self.canvas.img
            self._undo()
            X[i] = x[:, :, :1] / 255

        # [0, 1] - the higher the better reward
        model_output = self.model.predict(X)
        return -distances / model_output

    def step(self, shape):
        self.prev_img = self.canvas.img.copy()
        self.prev_distance = self.distance.copy()

        bounds = self.canvas.add(shape)

        self.update_distance(
            distance=self.distance,
            bounds=bounds,
            img=self.canvas.img,
            target=self.canvas.target
        )

        if self.save_actions:
            self._save_shape(shape)

        return self._distance()

    def _undo(self):
        self.canvas.img = self.prev_img
        self.distance = self.prev_distance
        if self.save_actions:
            self._remove_last_shape()

    def _distance(self):
        return np.average(self.distance)
