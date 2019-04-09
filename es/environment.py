import logging

import numpy as np

from es.reward import DistanceReward, Mixed

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

    def evaluate(self, shape, n):
        current_reward = self.reward_config[n]
        assert isinstance(current_reward, DistanceReward)
        bounds = self.inner_step(shape)
        reward = current_reward.get(bounds)
        self._undo()
        return reward

    def evaluate_batch(self, shapes, n):
        current_reward = self.reward_config[n]
        if isinstance(current_reward, DistanceReward):
            return [self.evaluate(shape, n) for shape in shapes]
        elif isinstance(current_reward, Mixed):
            img_shape = self.canvas.img.shape
            X = np.empty((len(shapes), img_shape[0], img_shape[1], img_shape[2]))
            B = []

            for i, shape in enumerate(shapes):
                bounds = self.inner_step(shape)
                x = self.canvas.img
                self._undo()
                X[i] = x
                B.append(bounds)

            params = {"X": X, "bounds": B}
            return current_reward.get(params)
        else:
            img_shape = self.canvas.img.shape
            X = np.empty((len(shapes), img_shape[0], img_shape[1], img_shape[2]))

            for i, shape in enumerate(shapes):
                self.inner_step(shape)
                x = self.canvas.img
                self._undo()
                X[i] = x

            return current_reward.get(X)

    def step(self, shape, n):
        bounds = self.canvas.add(shape)

        current_reward = self.reward_config[n]
        if isinstance(current_reward, DistanceReward):
            current_reward.update(bounds)

        if self.save_actions:
            self._save_shape(shape)

    def inner_step(self, shape):
        self.prev_img = self.canvas.img.copy()
        bounds = self.canvas.add(shape)
        return bounds

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

    def _undo(self):
        self.canvas.img = self.prev_img

    def _save_shape(self, shape):
        self.shapes[self.current_shape_num] = (
            shape.__class__,
            shape.normalized_params(*self.observation_shape())
        )
        self.current_shape_num += 1
