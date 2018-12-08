import logging

import matplotlib
import matplotlib.image as mimg

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target):
        self.target = mimg.imread(target) / 255
        self.img = np.ones(self.target.shape, dtype=np.float)
        logger.debug(f'Initialized canvas with target shape: {self.target.shape}')

    def show(self):
        plt.imshow(np.concatenate((self.target, self.img), axis=1))
        plt.waitforbuttonpress()
