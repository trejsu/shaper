import logging

import matplotlib
import matplotlib.image as mimg

from shaper.util import mse_full, average_color, update_mse

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target):
        self.target = mimg.imread(target)[:, :, :3].astype(np.float)
        self.img = np.full(self.target.shape, average_color(self.target), dtype=np.float)
        self.mse = mse_full(target=self.target, x=self.img)
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

    def _showable_img(self):
        return np.concatenate((self.target / 255, self.img / 255), axis=1)

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
        return self._score()

    def size(self):
        return self.img.shape[1], self.img.shape[0]

    def undo(self):
        self.img = self.prev_img
        self.mse = self.prev_mse

    def save(self, output):
        mimg.imsave(output, self.img.astype(np.uint8))

    def evaluate(self, shape):
        score = self.add(shape)
        self.undo()
        return score

    def _score(self):
        return np.average(self.mse)
