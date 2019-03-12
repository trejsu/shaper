import logging

import matplotlib
import matplotlib.image as mimg

from .util import average_color, resize_to_size, read_img, hex_to_rgb

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)


class Canvas(object):
    def __init__(self, target, size, background):
        target_img = self._get_target_img(target)
        self.target = resize_to_size(img=target_img, size=size).astype(np.float)

        self.background = background
        self.color = self._get_color(background)

        self.img = np.full(self.target.shape, self.color, dtype=np.float)
        assert self.target.shape == self.img.shape, 'Target and img must have the same shape'

        self.showed = None
        self.fig = None

        log.debug(f'Initialized canvas with target shape: {self.target.shape}')
        log.debug(f'Target min: {np.min(self.target)}, target max: {np.max(self.target)}')

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

        if target_img.shape[2] == 1:
            target_img = np.repeat(target_img, 3, axis=2)

        return target_img

    def clear_and_resize(self, size):
        return Canvas(target=self.target_path, size=size, background=self.background)

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

    def add(self, shape, color):
        return shape.render(self.img, self.target, color)

    def size(self):
        return self.img.shape[1], self.img.shape[0]

    def save(self, output):
        log.debug(f'Saving image under {output}')
        mimg.imsave(output, self.img.astype(np.uint8))

    def _showable_img(self):
        return np.concatenate((self.target / 255, self.img / 255), axis=1)
