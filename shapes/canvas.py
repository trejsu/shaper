import logging

import matplotlib
import matplotlib.image as mimg

from .util import average_color, resize_to_size, read_img, hex_to_rgb

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target, size, background, channels, from_target=False):
        assert channels in [1, 3]
        self.channels = channels

        target_img = self._get_target_img(target)
        self.target = resize_to_size(img=target_img, size=size).astype(np.float)
        assert self.target.shape[-1] == channels

        self.background = background
        self.color = self._get_color(background)

        if from_target:
            self.img = self.target.copy()
        else:
            self.img = np.full(self.target.shape, self.color, dtype=np.float)

        assert self.target.shape == self.img.shape, 'Target and img must have the same shape'

        self.showed = None
        self.fig = None

        log.debug(f'Initialized canvas with target shape: {self.target.shape}')
        log.debug(f'Target min: {np.min(self.target)}, target max: {np.max(self.target)}')

    @staticmethod
    def without_target(size, background, channels):
        target = np.empty((size, size, channels))
        return Canvas(target=target, size=size, background=background, channels=channels)

    def _get_color(self, background):
        return average_color(self.target) if background is None \
            else hex_to_rgb(background) if isinstance(background, str) \
            else background

    def _get_target_img(self, target):
        if isinstance(target, str):
            self.target_path = target
            target_img = read_img(target)
        else:
            target_img = target

        if len(target_img.shape) == 2:
            target_img = target_img.reshape(target_img.shape[0], target_img.shape[1], 1)

        if target_img.shape[2] != self.channels:
            if self.channels == 3:
                target_img = np.repeat(target_img, 3, axis=2)
            else:
                target_img = target_img[:, :, :1]

        if np.max(target_img) <= 1:
            target_img *= 255

        return target_img.astype(np.uint8)

    def clear_and_resize(self, size):
        return Canvas(target=self.target_path, size=size, background=self.background,
                      channels=self.channels)

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
        return shape.render(self.img, self.target)

    def size(self):
        return self.img.shape[1], self.img.shape[0]

    def save(self, output):
        log.debug(f'Saving image under {output}')
        img = self.img.copy()
        if self.img.shape[-1] == 1:
            img = np.repeat(self.img, 3, axis=2)
        mimg.imsave(output, img.astype(np.uint8))

    def reset(self, from_target=False):
        del self.img
        if from_target:
            self.img = self.target.copy()
        else:
            self.img = np.full(self.target.shape, self.color, dtype=np.float)

    def _showable_img(self):
        target = self.target.copy()
        img = self.img.copy()
        if self.target.shape[-1] == 1:
            target = np.repeat(self.target, 3, axis=2)
            img = np.repeat(self.img, 3, axis=2)
        return np.concatenate((target / 255, img / 255), axis=1)
