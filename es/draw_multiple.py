import logging
from collections import namedtuple

import numpy as np
from keras.datasets import mnist

from es.environment import Environment
from es.model import ModelA
from es.optimizer import GradientDescent, Adam, Momentum, Nesterov, Adadelta, Adagrad, RMSProp
from es.reward import L1, L2, MSE, Activation, Mixed
from es.strategy import RandomStrategy, EvolutionStrategy, SimpleEvolutionStrategy
from shapes.canvas import Canvas

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# todo: refactor to class
def draw(images, n, alpha=0.5, random=100, sample=10, step=100, learning_rate=4.64,
    sigma_factor=0.03, algorithm='natural', optimizer='adam', shape_mode=0, seed=None,
    rewards='mse', rewards_thresholds='1', rewards_coeffs='1e-6,1',
    scale_decay=0.00005, background=None, save_all=False, save_actions=False, channels=1):
    rng = np.random.RandomState(seed=seed if seed is not None else np.random.randint(0, 2 ** 32))

    if save_all:
        result = [np.empty(images.shape) for _ in range(n)]
    else:
        result = np.empty(images.shape)

    for idx in range(len(images)):
        env = init(
            input=images[idx],
            background=background,
            rewards=rewards,
            n=n,
            save_actions=save_actions,
            rewards_thresholds=rewards_thresholds,
            channels=channels,
            rewards_coeffs=rewards_coeffs
        )

        random_strategy = RandomStrategy(
            random,
            *env.observation_shape(),
            alpha=alpha,
            shape_mode=shape_mode,
            rng=rng,
            decay=scale_decay
        )

        for i in range(1, n + 1):
            best_score, best_shape = find_best_shape(env=env, strategy=random_strategy, action=i)

            strategy = pick_strategy(best_shape=best_shape, env=env, algorithm=algorithm,
                                     optimizer=optimizer,
                                     alpha=alpha, sample=sample, sigma_factor=sigma_factor,
                                     learning_rate=learning_rate,
                                     shape_mode=shape_mode, rng=rng)

            for j in range(1, step + 1):
                score, shape = find_best_shape(env=env, strategy=strategy, action=i)

                if score < best_score:
                    best_score = score
                    best_shape = shape

            env.step(shape=best_shape, n=n)
            if save_all:
                result[i - 1][idx] = extract_drawing(env)

        if not save_all:
            result[idx] = extract_drawing(env)

    return result


def extract_drawing(env):
    drawing = env.canvas.img
    if drawing.shape[2] == 3:
        drawing = drawing[:, :, :1]
    return drawing


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

    rewards_instances = {
        'mse': MSE(canvas),
        'l1': L1(canvas),
        'l2': L2(canvas),
        'conv1': Activation(canvas, ModelA, {"layer": ModelA.CONV1}),
        'conv2': Activation(canvas, ModelA, {"layer": ModelA.CONV2}),
        'dense': Activation(canvas, ModelA, {"layer": ModelA.DENSE}),
        'mse+conv1': Mixed(
            rewards=[MSE(canvas), Activation(canvas, ModelA, {"layer": ModelA.CONV1})],
            coeffs=coeffs
        )
    }

    reward_config = {}
    for a in range(1, config.n + 1):
        for t_idx, t in enumerate(thresholds):
            if a < t:
                reward_config[a] = rewards_instances[rewards[t_idx - 1]]
                break
        else:
            reward_config[a] = rewards_instances[rewards[-1]]

    assert len(reward_config) == config.n
    return reward_config


def init(input, background, rewards, n, save_actions, rewards_thresholds, channels, rewards_coeffs):
    canvas = Canvas(
        target=input,
        size=max(input.shape[0], input.shape[1]),
        background=background,
        channels=channels
    )

    Config = namedtuple('Config', ['n', 'rewards', 'rewards_thresholds', 'rewards_coeffs'])
    config = Config(
        n=n,
        rewards=rewards,
        rewards_thresholds=rewards_thresholds,
        rewards_coeffs=rewards_coeffs
    )

    reward_config = get_reward_config(canvas, config)

    env = Environment(
        canvas=canvas,
        reward_config=reward_config,
        num_shapes=n,
        save_actions=save_actions
    )

    return env


def pick_strategy(best_shape, env, algorithm, optimizer, alpha, sample, sigma_factor, learning_rate,
    shape_mode, rng):
    if algorithm == 'natural':
        optimizer = {
            'sgd': GradientDescent,
            'momentum': Momentum,
            'nesterov': Nesterov,
            'adagrad': Adagrad,
            'rmsprop': RMSProp,
            'adadelta': Adadelta,
            'adam': Adam
        }[optimizer]

        strategy = EvolutionStrategy(
            best_shape,
            *env.observation_shape(),
            alpha=alpha,
            n=sample,
            sigma_factor=sigma_factor,
            optimizer=optimizer(
                initial_params=best_shape.params(),
                learning_rate=learning_rate
            ),
            shape_mode=shape_mode,
            rng=rng
        )
    elif algorithm == 'simple':
        strategy = SimpleEvolutionStrategy(
            best_shape,
            *env.observation_shape(),
            alpha=alpha,
            n=sample,
            sigma_factor=sigma_factor,
            shape_mode=shape_mode,
            rng=rng
        )
    else:
        strategy = RandomStrategy(
            sample,
            *env.observation_shape(),
            alpha=alpha,
            rng=rng
        )
    return strategy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)

    args = parser.parse_args()


    def data_mnist():
        (_, _), (X_test, _) = mnist.load_data()
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255
        print("Loaded MNIST test data.")
        return X_test


    # def load(path):
    #     adv_samples_path = os.path.join(BASE, path)
    #     with np.load(adv_samples_path) as adv_samples:
    #         X = adv_samples['X']
    #         Y = adv_samples['Y']
    #         pred = adv_samples['pred']
    #         prob = adv_samples['prob']
    #         log.info(f'X.shape - {X.shape}')
    #         log.info(f'Y.shape - {Y.shape}')
    #         log.info(f'pred.shape - {pred.shape}')
    #         log.info(f'prob.shape - {prob.shape}')
    #     return X, Y, pred, prob

    X = data_mnist()
    X_redrawned = draw(images=X, n=100, alpha=0.8, background='000000', save_all=True)
    assert len(X_redrawned) == 100
    for n in range(1, 101):
        np.savez(args.output_path % n, targets=X, drawings=X_redrawned[n - 1])
