import logging
from collections import namedtuple

import numpy as np
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
