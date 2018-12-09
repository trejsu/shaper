import logging

import matplotlib
import matplotlib.image as mimg

from shaper.imgutils import mse_full, mse_partial

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target):
        self.target = mimg.imread(target)
        self.img = np.full(self.target.shape, 255, dtype=np.uint8)
        self.mse = mse_full(target=self.target, x=self.img)
        self.prev_img = None
        self.prev_mse = None
        self.showed = None
        log.debug(f'Initialized canvas with target shape: {self.target.shape}')
        log.debug(f'Target min: {np.min(self.target)}, target max: {np.max(self.target)}')
        assert self.target.shape == self.img.shape, 'Target and img must have the same shape'
        assert self.target.dtype == np.uint8, f'Unexpected target type: {self.target.dtype}'

    def init(self):
        return self._score()

    def show_and_wait(self):
        plt.imshow(np.concatenate((self.target, self.img), axis=1))
        plt.waitforbuttonpress()

    def show(self):
        img = np.concatenate((self.target, self.img), axis=1)
        plt.imshow(img)
        plt.show(block=False)

    def add(self, shape):
        self.prev_img = self.img.copy()
        self.prev_mse = self.mse
        changed = shape.render(self.img)
        changed_mse = mse_partial(self.target, self.img, changed)
        assert changed.shape == changed_mse.shape
        self.mse = np.where(changed == 0, self.mse, changed_mse)
        return self._score()

    def size(self):
        return self.img.shape[1], self.img.shape[0]

    def undo(self):
        self.img = self.prev_img.copy()
        self.mse = self.prev_mse

    def save(self, output):
        mimg.imsave(output, self.img)

    def _score(self):
        return np.average(self.mse)
