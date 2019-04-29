import numpy as np

from es.reward import MSE, L1, L2, Activation


def find_best_shape(env, strategy, action=None):
    shapes = strategy.ask() if action is None else strategy.ask(action)
    scores = env.evaluate_batch(shapes=shapes, n=action)
    strategy.tell(scores)
    shape, score = strategy.result()
    return score, shape


def get_reward_config(canvas, config):
    rewards = config.rewards.split(',')
    assert len(rewards) > 0

    thresholds = np.fromstring(config.rewards_thresholds, dtype=int, sep=',')
    assert len(thresholds) > 0
    assert thresholds[0] == 1

    coeffs = np.fromstring(config.rewards_coeffs, dtype=float, sep=',')

    scale = {
        'mse+conv1': lambda c: [c[0] * 1e-6, c[1]]
    }

    rewards_instances = {
        'mse': lambda: MSE(canvas),
        'l1': lambda: L1(canvas),
        'l2': lambda: L2(canvas),
        'conv1': conv1(canvas),
        'conv2': conv2(canvas),
        'dense': dense(canvas),
        # 'mse+conv1': lambda: Mixed(
        #     rewards=[MSE(canvas), conv1(canvas)],
        #     coeffs=scale['mse+conv1'](coeffs)
        # )
    }

    reward_config = {}
    for a in range(1, config.n + 1):
        for t_idx, t in enumerate(thresholds):
            if a < t:
                reward_config[a] = rewards_instances[rewards[t_idx - 1]]()
                break
        else:
            reward_config[a] = rewards_instances[rewards[-1]]()

    assert len(reward_config) == config.n
    return reward_config


def conv1(canvas):
    from es.model import ModelA
    return lambda: Activation(canvas, ModelA, {"layer": ModelA.CONV1})


def conv2(canvas):
    from es.model import ModelA
    return lambda: Activation(canvas, ModelA, {"layer": ModelA.CONV2})


def dense(canvas):
    from es.model import ModelA
    return lambda: Activation(canvas, ModelA, {"layer": ModelA.DENSE})
