import numpy as np
from PIL import Image


def resize(img, w, h):
    result = Image.fromarray(img)
    result = result.resize((w, h))
    return np.array(result)


def mse_full(target, x):
    return np.average(np.square(target - x))
