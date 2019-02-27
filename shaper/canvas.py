import logging

import matplotlib
import matplotlib.image as mimg

from .util import l2_full, average_color, update_l2, resize_to_size, update_l1, l1_full, read_img, hex_to_rgb

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target, size, num_shapes, metric, background, save_actions):
        self.target_path = target
        self.target = resize_to_size(img=read_img(target), size=size).astype(np.float)
        self.color = average_color(self.target) if background is None else hex_to_rgb(background)
        self.img = np.full(self.target.shape, self.color, dtype=np.float)
        self.distance = (l1_full if metric == 'l1' else l2_full)(target=self.target, x=self.img)
        self.update_distance = update_l1 if metric == 'l1' else update_l2
        self.save_actions = save_actions
        self.shapes = [None] * num_shapes
        self.current_shape_num = 0
        self.prev_img = None
        self.prev_distance = None
        self.showed = None
        self.fig = None
        log.debug(f'Initialized canvas with target shape: {self.target.shape}')
        log.debug(f'Target min: {np.min(self.target)}, target max: {np.max(self.target)}')
        assert self.target.shape == self.img.shape, 'Target and img must have the same shape'

    def init(self):
        return self._score()

    def show_and_wait(self):
        plt.imshow(self._showable_img())
        plt.waitforbuttonpress()

    def show(self):
        img = self._showable_img()
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure()
            self.showed = plt.imshow(img)
        else:
            self.showed.set_data(img)
            self.fig.canvas.draw()

    def add(self, shape, color=None):
        self.prev_img = self.img.copy()
        self.prev_distance = self.distance.copy()
        bounds = shape.render(self.img, self.target, color)
        self.update_distance(
            distance=self.distance,
            bounds=bounds,
            img=self.img,
            target=self.target
        )
        if self.save_actions:
            self._save_shape(shape)
        return self._score()

    def size(self):
        return self.img.shape[1], self.img.shape[0]

    def undo(self):
        self.img = self.prev_img
        self.distance = self.prev_distance
        if self.save_actions:
            self._remove_last_shape()

    def _remove_last_shape(self):
        self.current_shape_num -= 1

    def save_in_size(self, output, size):
        if self.save_actions:
            log.debug(f'Saving image under {output} in size = {size}')
            target = resize_to_size(
                img=read_img(self.target_path),
                size=size
            ).astype(np.float)

            img = np.full(
                shape=target.shape,
                fill_value=self.color,
                dtype=np.float
            )

            assert img.shape == target.shape

            for s in self.shapes:
                if s is None:
                    break
                else:
                    cls, normalized_params = s
                shape = cls.from_normalized_params(
                    img.shape[1],
                    img.shape[0],
                    *normalized_params
                )
                # todo: save in size will render with average colors even if shapes were given in specific ones
                shape.render(img=img, target=target, color=None)

            mimg.imsave(output, img.astype(np.uint8))
        else:
            raise Exception("Cannot save in size, save_actions set to False")

    def save(self, output):
        log.debug(f'Saving image under {output}')
        mimg.imsave(output, self.img.astype(np.uint8))

    def evaluate(self, shape):
        score = self.add(shape)
        self.undo()
        return score

    def _score(self):
        return np.average(self.distance)

    def _save_shape(self, shape):
        self.shapes[self.current_shape_num] = (
            shape.__class__,
            shape.normalized_params(*self.size()),
        )
        self.current_shape_num += 1

    def _showable_img(self):
        return np.concatenate((self.target / 255, self.img / 255), axis=1)
