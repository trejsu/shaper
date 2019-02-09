import logging

import matplotlib
import matplotlib.image as mimg

from shaper.util import l2_full, average_color, update_l2, resize_to_size, update_l1, l1_full, read_img
from .util import timeit

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target, size, output_size, num_shapes, metric):
        self.target_path = target
        self.target = resize_to_size(img=read_img(target), size=size).astype(np.float)
        self.img = np.full(self.target.shape, average_color(self.target), dtype=np.float)
        self.distance = (l1_full if metric == 'l1' else l2_full)(target=self.target, x=self.img)
        self.output_size = output_size
        self.update_distance = update_l1 if metric == 'l1' else update_l2
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

    @timeit
    def add(self, shape):
        self.prev_img = self.img.copy()
        self.prev_distance = self.distance.copy()
        bounds = shape.render(self.img, self.target)
        self.update_distance(
            distance=self.distance,
            bounds=bounds,
            img=self.img,
            target=self.target
        )
        self._add_to_list(shape)
        return self._score()

    @timeit
    def size(self):
        return self.img.shape[1], self.img.shape[0]

    @timeit
    def undo(self):
        self.img = self.prev_img
        self.distance = self.prev_distance
        self.current_shape_num -= 1

    @timeit
    def save(self, output):
        target = resize_to_size(
            img=read_img(self.target_path),
            size=self.output_size
        ).astype(np.float)

        img = np.full(
            shape=target.shape,
            fill_value=average_color(self.target),
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
            shape.render(img=img, target=target)

        mimg.imsave(output, img.astype(np.uint8))

    @timeit
    def save_nth(self, output):
        mimg.imsave(output, self.img.astype(np.uint8))

    @timeit
    def evaluate(self, shape):
        score = self.add(shape)
        self.undo()
        return score

    @timeit
    def _score(self):
        return np.average(self.distance)

    @timeit
    def _add_to_list(self, shape):
        self.shapes[self.current_shape_num] = (
            shape.__class__,
            shape.normalized_params(*self.size()),
        )
        self.current_shape_num += 1

    def _showable_img(self):
        return np.concatenate((self.target / 255, self.img / 255), axis=1)


