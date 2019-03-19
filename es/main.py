import logging
import os
import time

# todo: remove and find another solution
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from es.environment import DistanceEnv, NNEnv, MixedEnv
from es.optimizer import GradientDescent, Adam, Momentum, Nesterov, Adadelta, Adagrad, RMSProp
from es.strategy import RandomStrategy, EvolutionStrategy, SimpleEvolutionStrategy
from shapes.canvas import Canvas
from es.model import Classifier, Discriminator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main(_):
    env, show, save = init()
    start = time.time()

    random = RandomStrategy(
        args.random,
        *env.observation_shape(),
        alpha=args.alpha,
        shape_mode=args.shape_mode,
        rng=rng,
        decay=args.scale_decay
    )

    for i in tqdm(range(1, args.n + 1)):
        best_reward, best_shape = find_best_shape(env=env, strategy=random, action=i)

        strategy = pick_strategy(best_shape=best_shape, env=env)

        for j in range(1, args.step + 1):
            reward, shape = find_best_shape(env=env, strategy=strategy)

            if reward > best_reward:
                best_reward = reward
                best_shape = shape

        env.step(best_shape)
        tqdm.write(f'Action {i}, new reward: {best_reward:.4f}')

        show()

        if save_every_action():
            save(args.output % i)

    elapsed = time.time() - start
    shapes_drawn = args.n * (args.step * args.sample + args.random)
    log.info(f'Total shapes drawn {shapes_drawn}, time {elapsed:.2f} s, '
             f'({shapes_drawn / elapsed:.1f} shapes/s)')

    if save_final():
        save(args.output)


def find_best_shape(env, strategy, action=None):
    shapes = strategy.ask() if action is None else strategy.ask(action=action)
    scores = env.evaluate_batch(shapes)
    strategy.tell(scores)
    shape, score = strategy.result()
    return score, shape


def init():
    canvas = Canvas(
        target=args.input,
        size=args.resize,
        background=args.background
    )

    show = show_function(canvas)
    env = pick_environment(canvas)
    save = save_function(canvas, env)

    return env, show, save


def pick_environment(canvas):
    def classifier():
        return Classifier(args.label)

    def discriminator():
        return Discriminator(args.label)

    return {
        'distance': DistanceEnv(
            canvas=canvas,
            metric=args.metric,
            num_shapes=args.n,
            save_actions=args.save_actions
        ),
        'C': NNEnv(
            canvas=canvas,
            num_shapes=args.n,
            save_actions=args.save_actions,
            model_initializer=classifier
        ),
        'mixed-C': MixedEnv(
            canvas=canvas,
            metric=args.metric,
            num_shapes=args.n,
            save_actions=args.save_actions,
            model_initializer=classifier
        ),
        'D': NNEnv(
            canvas=canvas,
            num_shapes=args.n,
            save_actions=args.save_actions,
            model_initializer=discriminator
        ),
        'mixed-D': MixedEnv(
            canvas=canvas,
            metric=args.metric,
            num_shapes=args.n,
            save_actions=args.save_actions,
            model_initializer=discriminator
        )
    }[args.env]


def pick_strategy(best_shape, env):
    if args.algorithm == 'natural':
        optimizer = {
            'sgd': GradientDescent,
            'momentum': Momentum,
            'nesterov': Nesterov,
            'adagrad': Adagrad,
            'rmsprop': RMSProp,
            'adadelta': Adadelta,
            'adam': Adam
        }[args.optimizer]

        strategy = EvolutionStrategy(
            best_shape,
            *env.observation_shape(),
            alpha=args.alpha,
            n=args.sample,
            sigma_factor=args.sigma_factor,
            optimizer=optimizer(
                initial_params=best_shape.params(),
                learning_rate=args.learning_rate
            ),
            shape_mode=args.shape_mode,
            rng=rng
        )
    elif args.algorithm == 'simple':
        strategy = SimpleEvolutionStrategy(
            best_shape,
            *env.observation_shape(),
            alpha=args.alpha,
            n=args.sample,
            sigma_factor=args.sigma_factor,
            shape_mode=args.shape_mode,
            rng=rng
        )
    else:
        strategy = RandomStrategy(
            args.sample,
            *env.observation_shape(),
            alpha=args.alpha,
            rng=rng
        )
    return strategy


def show_function(canvas):
    return canvas.show_and_wait if args.render_mode == 0 else canvas.show if args.render_mode == 1 else lambda: None


def save_function(canvas, env):
    return (lambda output: env.save_in_size(output,
                                            args.output_size)) if args.save_actions else canvas.save


def save_every_action():
    return args.output is not None and '%d' in args.output


def save_final():
    return args.output is not None and '%d' not in args.output


if __name__ == '__main__':
    flags = tf.app.flags

    flags.DEFINE_integer('n', None, 'Number of triangles to draw')
    flags.DEFINE_string('input', None, 'Target image path')
    flags.DEFINE_string('output', 'out.jpg', 'Output image path')
    flags.DEFINE_integer('render_mode', 2, 'Render mode: 0 - click, 1 - automatic, 2 - no render')
    flags.DEFINE_float('alpha', 0.5, 'Alpha value [0, 1]')
    flags.DEFINE_integer('random', 100, '')
    flags.DEFINE_integer('sample', 10, '')
    flags.DEFINE_integer('step', 100, '')
    flags.DEFINE_float('learning_rate', 4.64, '')
    flags.DEFINE_float('sigma_factor', 0.03, '')
    flags.DEFINE_string('algorithm', 'natural', '')
    flags.DEFINE_string('optimizer', 'adam', '')
    flags.DEFINE_integer('shape_mode', 0, '')
    flags.DEFINE_integer('resize', 100, 'Size to which input will be scaled before drawing - the '
                                        'bigger the longer it will take but the more details can '
                                        'be captured')
    flags.DEFINE_integer('output_size', 512, 'Output image size')
    flags.DEFINE_boolean('save_actions', False, 'When not present, output-size parameter will be '
                                                'ignored, and output image will be saved in size '
                                                'equal to resize parameter')
    flags.DEFINE_integer('seed', None, '')
    flags.DEFINE_string('metric', 'l2', '')
    flags.DEFINE_float('scale_decay', 0.00005, '')
    flags.DEFINE_string('background', None, 'Initial background color (hex value without #), if '
                                            'not passed, will be the average target img color')
    flags.DEFINE_integer('label', None, '')
    flags.DEFINE_string('env', 'distance', 'Environment: distance - reward based on mse, '
                                           'C - reward based on classifier output '
                                           'D - reward based on discriminator output')

    args = tf.app.flags.FLAGS

    seed = args.seed if args.seed is not None else np.random.randint(0, 2 ** 32)
    rng = np.random.RandomState(seed=seed)
    log.info(f'Rng seed = {seed}')

    try:
        tf.app.run()
    except Exception as e:
        logging.exception("")
