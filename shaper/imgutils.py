import numpy as np
from PIL import Image
from numba import njit


def resize(img, w, h):
    result = Image.fromarray(img)
    result = result.resize((w, h))
    return np.array(result)


def mse_full(target, x):
    return np.square(target - x)


def mse_partial(target, x, mask):
    return np.where(mask == 0, np.zeros(target.shape), np.square(target - x))


def average_color(img):
    return np.average(img, axis=(0, 1))


@njit("(f8[:,:,:], i8[:,:], f8[:,:,:], f8[:,:,:])")
def update_mse(mse, bounds, img, target):
    for i in range(len(bounds)):
        x1 = bounds[i, 0]
        x2 = bounds[i, 1]
        y = bounds[i, 2]
        direction = 1 if x1 < x2 else -1
        mse[y, x1: x2 + 1: direction] = np.square(
            target[y, x1: x2 + 1: direction] - img[y, x1: x2 + 1: direction])
