import logging

import matplotlib
import matplotlib.image as mimg

from shaper.imgutils import mse_full, average_color

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
        log.debug(f'Initialized canvas with target shape: {self.target.shape}')
        log.debug(f'Target min: {np.min(self.target)}, target max: {np.max(self.target)}')
        assert self.target.shape == self.img.shape, 'Target and img must have the same shape'

    def init(self):
        return self._score()

    def show_and_wait(self):
        plt.imshow(np.concatenate((self.target / 255, self.img / 255), axis=1))
        plt.waitforbuttonpress()

    def show(self):
        img = np.concatenate((self.target / 255, self.img / 255), axis=1)
        plt.imshow(img)
        plt.show(block=False)

    def add(self, shape):
        self.prev_img = self.img.copy()
        self.prev_mse = self.mse
        shape.render(self.img)
        self.mse = mse_full(self.target, self.img)
        return self._score()

    def size(self):
        return self.img.shape[1], self.img.shape[0]

    def undo(self):
        self.img = self.prev_img.copy()
        self.mse = self.prev_mse

    def save(self, output):
        mimg.imsave(output, self.img.astype(np.uint8))

    def _score(self):
        return np.average(self.mse)
