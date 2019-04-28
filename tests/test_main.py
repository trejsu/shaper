from collections import namedtuple

import numpy as np

from es.main import get_reward_config
from es.reward import MSE, L2, L1
from shapes.canvas import Canvas

Config = namedtuple('Config', ['n', 'rewards', 'rewards_thresholds', 'rewards_coeffs'],
                    verbose=True)
canvas = Canvas(target=np.empty((100, 100)), size=100, background=None, channels=1)


def test_get_reward_config_for_single_reward_type():
    n = 10
    config = Config(n=n, rewards='mse', rewards_thresholds='1', rewards_coeffs='1,1')

    reward_config = get_reward_config(
        canvas=canvas,
        config=config
    )

    assert len(reward_config) == n
    for i in range(1, 11):
        assert isinstance(reward_config[i], MSE)


def test_get_reward_config_for_double_reward_type():
    n = 20
    config = Config(n=n, rewards='mse,l2', rewards_thresholds='1,10', rewards_coeffs='1,1')

    reward_config = get_reward_config(
        canvas=canvas,
        config=config
    )

    assert len(reward_config) == n
    for i in range(1, 9):
        assert isinstance(reward_config[i], MSE)
    for i in range(10, 21):
        assert isinstance(reward_config[i], L2)


def test_get_reward_config_for_triple_reward_type():
    n = 20
    config = Config(n=n, rewards='mse,l2,l1', rewards_thresholds='1,10,15', rewards_coeffs='1,1')

    reward_config = get_reward_config(
        canvas=canvas,
        config=config
    )

    assert len(reward_config) == n
    for i in range(1, 10):
        assert isinstance(reward_config[i], MSE)
    for i in range(10, 15):
        assert isinstance(reward_config[i], L2)
    for i in range(15, 21):
        assert isinstance(reward_config[i], L1)
