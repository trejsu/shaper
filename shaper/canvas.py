import logging

import matplotlib
import matplotlib.image as mimg

from shaper.util import mse_full, average_color, update_mse, resize_to_size

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target, size, output_size, num_shapes):
        self.target = resize_to_size(img=mimg.imread(target)[:, :, :3], size=size).astype(np.float)
        self.img = np.full(self.target.shape, average_color(self.target), dtype=np.float)
        self.mse = mse_full(target=self.target, x=self.img)
        self.output_size = output_size
        self.shapes = np.empty(shape=(num_shapes, 2))
        self.current_shape_num = 0
        self.prev_img = None
        self.prev_mse = None
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

    def add(self, shape):
        self.prev_img = self.img.copy()
        self.prev_mse = self.mse.copy()
        bounds = shape.render(self.img, self.target)
        update_mse(mse=self.mse, bounds=bounds, img=self.img, target=self.target)
        self._add_to_list(shape)
        return self._score()

    def size(self):
        return self.img.shape[1], self.img.shape[0]

    def undo(self):
        self.img = self.prev_img
        self.mse = self.prev_mse
        self.current_shape_num -= 1

    # todo: save in requested output size
    def save(self, output):
        mimg.imsave(output, self.img.astype(np.uint8))

    def evaluate(self, shape):
        score = self.add(shape)
        self.undo()
        return score

    def _score(self):
        return np.average(self.mse)

    def _add_to_list(self, shape):
        self.shapes[self.current_shape_num] = [
            shape.__class__,
            shape.normalized_params(*self.size())
        ]
        self.current_shape_num += 1

    def _showable_img(self):
        return np.concatenate((self.target / 255, self.img / 255), axis=1)
