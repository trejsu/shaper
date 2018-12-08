import logging

import matplotlib
import matplotlib.image as mimg

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target):
        self.target = mimg.imread(target)
        self.img = np.full(self.target.shape, 255, dtype=np.uint8)
        log.debug(f'Initialized canvas with target shape: {self.target.shape}')
        log.debug(f'Target min: {np.min(self.target)}, target max: {np.max(self.target)}')
        assert self.target.dtype == np.uint8, f'Unexpected target type: {self.target.dtype}'

    def show(self):
        plt.imshow(np.concatenate((self.target, self.img), axis=1))
        plt.waitforbuttonpress()

    def add(self, shape):
        log.debug(f'Rendering {shape}')
        shape.render(self.img)

    def size(self):
        return self.img.shape[1], self.img.shape[0]
