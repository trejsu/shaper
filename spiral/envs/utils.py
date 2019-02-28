import numpy as np


def uniform_locations(screen_size, location_size, object_radius, normalize=False):
    x = np.linspace(object_radius, screen_size - object_radius, location_size)
    grid = np.meshgrid(x, x)
    out = np.array(zip(*np.vstack(map(np.ravel, grid))))
    if normalize:
        div = location_size ** 2 / 2
        out = (out - div) / div
    return out
