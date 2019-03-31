import time
from collections import defaultdict

import matplotlib.image as mimg
import numpy as np
from PIL import Image
from numba import njit

MIN_VALUE = -9999999999999

times = defaultdict(int)
calls = defaultdict(int)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        name = f'{method.__module__}.{method.__name__}'
        times[name] += (te - ts) * 1000
        calls[name] += 1
        return result

    return timed


def resize_to_size(img, size):
    w = img.shape[1]
    h = img.shape[0]

    if w == size and h == size:
        return img

    if w > h:
        scale = size / w
    else:
        scale = size / h
    new_w = int(w * scale)
    new_h = int(h * scale)

    return resize(img=img, w=new_w, h=new_h)


def resize(img, w, h):
    result = Image.fromarray(img)
    result = result.resize((w, h), Image.ANTIALIAS)
    return np.array(result)


def mse_full(target, x):
    return np.square(target - x)


def l1_full(target, x):
    return np.abs(target - x)


def average_color(img):
    return np.average(img, axis=(0, 1))


@njit("(f8[:,:,:], i8[:,:], f8[:,:,:], f8[:,:,:])")
def update_mse(distance, bounds, img, target):
    for i in range(len(bounds)):
        x1 = bounds[i, 0]
        x2 = bounds[i, 1]
        y = bounds[i, 2]
        distance[y, min(x1, x2): max(x1, x2) + 1] = np.square(
            target[y, min(x1, x2): max(x1, x2) + 1] - img[y, min(x1, x2): max(x1, x2) + 1])


@njit("(f8[:,:,:], i8[:,:], f8[:,:,:], f8[:,:,:])")
def update_l1(distance, bounds, img, target):
    for i in range(len(bounds)):
        x1 = bounds[i, 0]
        x2 = bounds[i, 1]
        y = bounds[i, 2]
        distance[y, min(x1, x2): max(x1, x2) + 1] = np.abs(
            target[y, min(x1, x2): max(x1, x2) + 1] - img[y, min(x1, x2): max(x1, x2) + 1])


def stardardize(x):
    x_minus_mean = np.array(x) - np.mean(x)
    return x_minus_mean if np.all(x_minus_mean == 0) else x_minus_mean / np.std(x)


def normalize(x):
    norm = np.sqrt((np.square(x)).sum())
    return x if norm == 0 else x / norm


@njit("i8[:,:](i8[:,:])")
def bounds_to_pixels(bounds):
    n = 0
    for i in range(len(bounds)):
        n += abs(bounds[i][0] - bounds[i][1]) + 1

    pixels = np.empty(shape=(n, 2), dtype=np.int64)

    j = 0
    for i in range(len(bounds)):
        for x in range(min(bounds[i, 0], bounds[i, 1]), max(bounds[i, 0], bounds[i, 1]) + 1):
            pixels[j, 0] = x
            pixels[j, 1] = bounds[i, 2]
            j += 1

    return pixels


def print_times():
    import operator
    mean_times = {name: t / calls[name] for name, t in times.items()}
    sorted_times = sorted(times.items(), key=operator.itemgetter(1))
    for name, t in sorted_times:
        print(f'{name}: {t:.2f}ms (calls = {calls[name]}, mean = {mean_times[name]:.2f}ms)')


def read_img(path):
    img = mimg.imread(path)
    if np.all(img <= 1):
        img *= 255
        img = img.astype('uint8')
    return img[:, :, :3]


def hex_to_rgb(hex):
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))
