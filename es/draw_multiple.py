import logging
from collections import namedtuple

import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

from es.draw_utils import find_best_shape, get_reward_config
from es.environment import Environment
from es.optimizer import GradientDescent, Adam, Momentum, Nesterov, Adadelta, Adagrad, RMSProp
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

    for idx in tqdm(range(len(images))):
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
                result[i - 1][idx] = env.canvas.img

        if not save_all:
            result[idx] = env.canvas.img

    return result


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
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--coeffs", type=str, required=True)

    args = parser.parse_args()

    NUM_SAMPLES = 1000


    def data_mnist():
        (_, _), (X_test, Y_test) = mnist.load_data()
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255
        print("Loaded MNIST test data.")
        return X_test[0:NUM_SAMPLES], Y_test[0:NUM_SAMPLES]


    X, Y = data_mnist()
    N = 20
    X_redrawned = draw(images=X, n=N, alpha=0.7, background='00', save_all=True,
                       rewards=args.reward, rewards_coeffs=args.coeffs)
    assert len(X_redrawned) == N
    for n in range(1, N + 1):
        np.savez(args.output_path % n, targets=X, drawings=X_redrawned[n - 1], Y=Y)
