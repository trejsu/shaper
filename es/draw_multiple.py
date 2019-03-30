import logging

import numpy as np
from tqdm import tqdm

from es.environment import Environment
from es.optimizer import GradientDescent, Adam, Momentum, Nesterov, Adadelta, Adagrad, RMSProp
from es.strategy import RandomStrategy, EvolutionStrategy, SimpleEvolutionStrategy
from shapes.canvas import Canvas

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def draw(images, n, alpha=0.5, random=100, sample=10, step=100, learning_rate=4.64, sigma_factor=0.03,
         algorithm='natural', optimizer='adam', shape_mode=0, seed=None, metric='l2', scale_decay=0.00005,
         background=None):
    rng = np.random.RandomState(seed=seed if seed is not None else np.random.randint(0, 2 ** 32))

    result = np.empty(images.shape)

    for idx, image in tqdm(enumerate(images)):
        env = init(input=image, background=background, metric=metric, n=n)

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

            strategy = pick_strategy(best_shape=best_shape, env=env, algorithm=algorithm, optimizer=optimizer,
                                     alpha=alpha, sample=sample, sigma_factor=sigma_factor, learning_rate=learning_rate,
                                     shape_mode=shape_mode, rng=rng)

            for j in range(1, step + 1):
                score, shape = find_best_shape(env=env, strategy=strategy)

                if score < best_score:
                    best_score = score
                    best_shape = shape

            _ = env.step(best_shape)
            # env.canvas.show_and_wait()

        drawing = env.canvas.img
        if drawing.shape[2] == 3:
            # print('-----')
            # print(drawing[:, :, 0][10])
            # print(drawing[:, :, 1][10])
            # print(drawing[:, :, 2][10])
            # print('-----')
            drawing = drawing[:, :, :1]
        result[idx] = drawing

    return result


def find_best_shape(env, strategy, action=None):
    shapes = strategy.ask() if action is None else strategy.ask(action=action)
    scores = [env.evaluate(shape) for shape in shapes]
    strategy.tell(scores)
    shape, score = strategy.result()
    return score, shape


def init(input, background, metric, n):
    canvas = Canvas(
        target=input,
        size=max(input.shape[0], input.shape[1]),
        background=background
    )

    env = Environment(
        canvas=canvas,
        metric=metric,
        num_shapes=n,
        save_actions=False
    )
    _ = env.init()

    return env


def pick_strategy(best_shape, env, algorithm, optimizer, alpha, sample, sigma_factor, learning_rate, shape_mode, rng):
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
