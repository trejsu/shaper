import numpy as np


def uniform_locations(screen_size, location_size, normalize=False):
    x = np.linspace(0, screen_size, location_size)
    grid = np.meshgrid(x, x)
    arrays = [
        grid[0].reshape(location_size, location_size, 1),
        grid[1].reshape(location_size, location_size, 1)
    ]
    out = np.concatenate(arrays, axis=2).reshape(-1, 2)
    if normalize:
        div = location_size ** 2 / 2
        out = (out - div) / div
    return out


def l2_dist(x, y):
    return np.sqrt(np.sum(np.square(x - y)))
