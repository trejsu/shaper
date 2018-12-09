import numpy as np
from PIL import Image


def resize(img, w, h):
    result = Image.fromarray(img)
    result = result.resize((w, h))
    return np.array(result)


def mse_full(target, x):
    return np.square(target - x)


def mse_partial(target, x, mask):
    return np.where(mask == 0, np.zeros(target.shape), np.square(target - x))
