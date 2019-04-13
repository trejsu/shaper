import logging
import os
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


class Drawer(object):
    def __init__(self, alpha=0.5, random=100, sample=10, step=100, learning_rate=4.64,
        sigma_factor=0.03, algorithm='natural', optimizer='adam', shape_mode=0, seed=None,
        rewards='mse', rewards_thresholds='1', rewards_coeffs='1e-6,1',
        scale_decay=0.00005, background=None, save_all=False, save_actions=False, channels=1):
        self.alpha = alpha
        self.random = random
        self.sample = sample
        self.step = step
        self.learning_rate = learning_rate
        self.sigma_factor = sigma_factor
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.shape_mode = shape_mode
        self.seed = seed
        self.rewards = rewards
        self.rewards_thresholds = rewards_thresholds
        self.rewards_coeffs = rewards_coeffs
        self.scale_decay = scale_decay
        self.background = background
        self.save_all = save_all
        self.save_actions = save_actions
        self.channels = channels
        self.rng = np.random.RandomState(
            seed=self.seed if self.seed is not None else np.random.randint(0, 2 ** 32))

    def draw(self, images, n):

        if self.save_all:
            result = [np.empty(images.shape) for _ in range(n)]
        else:
            result = np.empty(images.shape)

        for idx in tqdm(range(len(images))):
            env = self.init(input=images[idx], n=n)

            random_strategy = RandomStrategy(
                self.random,
                *env.observation_shape(),
                alpha=self.alpha,
                shape_mode=self.shape_mode,
                rng=self.rng,
                decay=self.scale_decay
            )

            for i in range(1, n + 1):
                best_score, best_shape = find_best_shape(env=env, strategy=random_strategy,
                                                         action=i)

                strategy = self.pick_strategy(best_shape=best_shape, env=env)

                for j in range(1, self.step + 1):
                    score, shape = find_best_shape(env=env, strategy=strategy, action=i)

                    if score < best_score:
                        best_score = score
                        best_shape = shape

                env.step(shape=best_shape, n=n)
                if self.save_all:
                    result[i - 1][idx] = env.canvas.img

            if not self.save_all:
                result[idx] = env.canvas.img

        return result

    def init(self, input, n):
        canvas = Canvas(
            target=input,
            size=max(input.shape[0], input.shape[1]),
            background=self.background,
            channels=self.channels
        )

        Config = namedtuple('Config', ['n', 'rewards', 'rewards_thresholds', 'rewards_coeffs'])
        config = Config(
            n=n,
            rewards=self.rewards,
            rewards_thresholds=self.rewards_thresholds,
            rewards_coeffs=self.rewards_coeffs
        )

        reward_config = get_reward_config(canvas, config)

        env = Environment(
            canvas=canvas,
            reward_config=reward_config,
            num_shapes=n,
            save_actions=self.save_actions
        )

        return env

    def pick_strategy(self, best_shape, env):
        if self.algorithm == 'natural':
            optimizer = {
                'sgd': GradientDescent,
                'momentum': Momentum,
                'nesterov': Nesterov,
                'adagrad': Adagrad,
                'rmsprop': RMSProp,
                'adadelta': Adadelta,
                'adam': Adam
            }[self.optimizer]

            strategy = EvolutionStrategy(
                best_shape,
                *env.observation_shape(),
                alpha=self.alpha,
                n=self.sample,
                sigma_factor=self.sigma_factor,
                optimizer=optimizer(
                    initial_params=best_shape.params(),
                    learning_rate=self.learning_rate
                ),
                shape_mode=self.shape_mode,
                rng=self.rng
            )
        elif self.algorithm == 'simple':
            strategy = SimpleEvolutionStrategy(
                best_shape,
                *env.observation_shape(),
                alpha=self.alpha,
                n=self.sample,
                sigma_factor=self.sigma_factor,
                shape_mode=self.shape_mode,
                rng=self.rng
            )
        else:
            strategy = RandomStrategy(
                self.sample,
                *env.observation_shape(),
                alpha=self.alpha,
                rng=self.rng
            )
        return strategy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--coeffs", type=str, required=True)
    parser.add_argument("--part-size", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1000)

    args = parser.parse_args()


    def data_mnist():
        (_, _), (X_test, Y_test) = mnist.load_data()
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255
        print("Loaded MNIST test data.")
        return X_test[0:args.num_samples], Y_test[0:args.num_samples]


    X, Y = data_mnist()
    N = 20

    drawer = Drawer(
        alpha=0.7,
        background='00',
        save_all=True,
        rewards=args.reward,
        rewards_coeffs=args.coeffs
    )

    num_parts = args.num_samples // args.part_size

    for part in tqdm(range(num_parts)):
        part_start = part * args.part_size
        part_end = part_start + args.part_size

        part_X = X[part_start:part_end]
        part_Y = Y[part_start:part_end]

        part_X_drawings = drawer.draw(images=part_X, n=N)
        assert len(part_X_drawings) == N

        for n in range(1, N + 1):
            np.savez(
                args.output_path % n + f'-part-{part}',
                targets=part_X,
                drawings=part_X_drawings[n - 1],
                Y=part_Y
            )

    print('Drawing complete.')
    print('Merging parts...')

    for n in range(1, N + 1):
        targets = np.empty((args.num_samples, 28, 28, 1))
        drawings = np.empty((args.num_samples, 28, 28, 1))
        Y = np.empty((args.num_samples,))

        for part in range(num_parts):
            part_start = part * args.part_size
            part_end = part_start + args.part_size

            part_path = (args.output_path % n) + f'-part-{part}.npz'
            part_data = np.load(part_path)
            targets[part_start:part_end] = part_data['targets']
            drawings[part_start:part_end] = part_data['drawings']
            Y[part_start:part_end] = part_data['Y']

        np.savez(args.output_path % n, targets=targets, drawings=drawings, Y=Y)

        for part in range(num_parts):
            part_path = (args.output_path % n) + f'-part-{part}.npz'
            os.remove(part_path)
