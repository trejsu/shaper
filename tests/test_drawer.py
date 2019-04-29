import numpy as np

from es.drawer import Drawer
from shapes.shape import from_shape_mode

IMAGES = np.empty((100, 28, 28, 1))
N = 100
SHAPE_MODE = 1
PARAMS_LEN = from_shape_mode(SHAPE_MODE).params_len()


def test_init_result_representation_save_all():
    drawer = Drawer(representation=True, save_all=True, shape_mode=SHAPE_MODE)
    result = drawer.initialize_result(images=IMAGES, n=N)
    assert len(result) == N
    assert all([r.shape == (IMAGES.shape[0], N, PARAMS_LEN) for r in result])


def test_init_result_representation_save_final():
    drawer = Drawer(representation=True, save_all=False, shape_mode=SHAPE_MODE)
    result = drawer.initialize_result(images=IMAGES, n=N)
    assert result.shape == (IMAGES.shape[0], N, PARAMS_LEN)


def test_init_result_imgs_save_all():
    drawer = Drawer(representation=False, save_all=True, shape_mode=SHAPE_MODE)
    result = drawer.initialize_result(images=IMAGES, n=N)
    assert len(result) == N
    assert all([r.shape == IMAGES.shape for r in result])


def test_init_result_imgs_save_final():
    drawer = Drawer(representation=False, save_all=False, shape_mode=SHAPE_MODE)
    result = drawer.initialize_result(images=IMAGES, n=N)
    assert result.shape == IMAGES.shape
