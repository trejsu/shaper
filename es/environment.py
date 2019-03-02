import logging

import numpy as np

from shapes.util import l2_full, l1_full, update_l1, update_l2

log = logging.getLogger(__name__)


class Environment(object):
    def __init__(self, canvas, metric, save_actions, num_shapes):
        self.canvas = canvas
        self.metric = metric
        self.save_actions = save_actions

        self.distance = (l1_full if metric == 'l1' else l2_full)(target=canvas.target, x=canvas.img)
        self.update_distance = update_l1 if metric == 'l1' else update_l2

        self.current_shape_num = 0
        self.shapes = [None] * num_shapes
        self.prev_img = None
        self.prev_distance = None

    def init(self):
        return self._score()

    def observation_shape(self):
        return self.canvas.size()

    def evaluate(self, shape, color=None):
        score = self.step(shape, color)
        self._undo()
        return score

    def step(self, shape, color=None):
        self.prev_img = self.canvas.img.copy()
        self.prev_distance = self.distance.copy()

        bounds = self.canvas.add(shape, color)

        self.update_distance(
            distance=self.distance,
            bounds=bounds,
            img=self.canvas.img,
            target=self.canvas.target
        )

        if self.save_actions:
            self._save_shape(shape, color)

        return self._score()

    def save_in_size(self, output, size):
        if self.save_actions:
            log.debug(f'Saving image under {output} in size = {size}')

            canvas = self.canvas.clear_and_resize(size=size)
            w, h = canvas.size()

            for s in self.shapes:
                if s is None:
                    break
                else:
                    cls, normalized_params, color = s

                shape = cls.from_normalized_params(w, h, *normalized_params)
                canvas.add(shape=shape, color=color)

            canvas.save(output=output)
        else:
            raise Exception("Cannot save in size, save_actions set to False")

    def _score(self):
        return np.average(self.distance)

    def _undo(self):
        self.canvas.img = self.prev_img
        self.distance = self.prev_distance
        if self.save_actions:
            self._remove_last_shape()

    def _remove_last_shape(self):
        self.current_shape_num -= 1

    def _save_shape(self, shape, color):
        self.shapes[self.current_shape_num] = (
            shape.__class__,
            shape.normalized_params(*self.observation_shape()),
            color
        )
        self.current_shape_num += 1
