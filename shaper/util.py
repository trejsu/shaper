import time
from collections import defaultdict

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


@timeit
def resize_to_size(img, size):
    w = img.shape[1]
    h = img.shape[0]
    if w > h:
        scale = size / w
    else:
        scale = size / h
    new_w = int(w * scale)
    new_h = int(h * scale)
    return resize(img=img, w=new_w, h=new_h)


@timeit
def resize(img, w, h):
    result = Image.fromarray(img)
    result = result.resize((w, h), Image.ANTIALIAS)
    return np.array(result)


@timeit
def l2_full(target, x):
    return np.square(target - x)


@timeit
def l1_full(target, x):
    return np.abs(target - x)


@timeit
def average_color(img):
    return np.average(img, axis=(0, 1))


@njit("(f8[:,:,:], i8[:,:], f8[:,:,:], f8[:,:,:])")
def update_l2(distance, bounds, img, target):
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


@timeit
def normalize(arr):
    arr_minus_mean = np.array(arr) - np.mean(arr)
    if np.all(arr_minus_mean == 0):
        return np.zeros(arr_minus_mean.shape)
    return arr_minus_mean / np.std(arr)


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
