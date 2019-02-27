import argparse
import logging
import time

import numpy as np

from es.optimizer import GradientDescent, Adam, Momentum, Nesterov, Adadelta, Adagrad, RMSProp
from es.strategy import RandomStrategy, EvolutionStrategy, SimpleEvolutionStrategy
from shaper.canvas import Canvas

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    canvas, show, save = init()
    start = time.time()

    random = RandomStrategy(
        ARGS.random,
        *canvas.size(),
        alpha=ARGS.alpha,
        shape_mode=ARGS.shape_mode,
        rng=ARGS.rng,
        decay=ARGS.scale_decay
    )

    for i in range(1, ARGS.n + 1):
        best_score, best_shape = find_best_shape(canvas=canvas, strategy=random, action=i)

        strategy = pick_strategy(best_shape, canvas)

        for j in range(1, ARGS.step + 1):
            score, shape = find_best_shape(canvas, strategy)

            if score < best_score:
                best_score = score
                best_shape = shape

        score = canvas.add(best_shape)
        log.info(f'Action {i}, new score: {score:.4f}')

        show()

        if save_every_action():
            save(ARGS.output % i)

    elapsed = time.time() - start
    shapes_drawn = ARGS.n * (ARGS.step * ARGS.sample + ARGS.random)
    log.info(f'Total shapes drawn {shapes_drawn}, time {elapsed:.2f} s, '
             f'({shapes_drawn / elapsed:.1f} shapes/s)')

    if save_final():
        save(ARGS.output)


def find_best_shape(canvas, strategy, action=None):
    shapes = strategy.ask() if action is None else strategy.ask(action=action)
    scores = [canvas.evaluate(shape) for shape in shapes]
    strategy.tell(scores)
    shape, score = strategy.result()
    return score, shape


def init():
    canvas = Canvas(
        target=ARGS.input,
        size=ARGS.resize,
        save_actions=ARGS.save_actions,
        num_shapes=ARGS.n,
        metric=ARGS.metric,
        background=ARGS.background
    )
    show = show_function(canvas)
    save = save_function(canvas)
    score = canvas.init()
    log.info(f'Initial score: {score}')
    return canvas, show, save


def pick_strategy(best_shape, canvas):
    if ARGS.algorithm == 'natural':
        optimizer = {
            'sgd': GradientDescent,
            'momentum': Momentum,
            'nesterov': Nesterov,
            'adagrad': Adagrad,
            'rmsprop': RMSProp,
            'adadelta': Adadelta,
            'adam': Adam
        }[ARGS.optimizer]

        strategy = EvolutionStrategy(
            best_shape,
            *canvas.size(),
            alpha=ARGS.alpha,
            n=ARGS.sample,
            sigma_factor=ARGS.sigma_factor,
            optimizer=optimizer(
                initial_params=best_shape.params(),
                learning_rate=ARGS.learning_rate
            ),
            shape_mode=ARGS.shape_mode,
            rng=ARGS.rng
        )
    elif ARGS.algorithm == 'simple':
        strategy = SimpleEvolutionStrategy(
            best_shape,
            *canvas.size(),
            alpha=ARGS.alpha,
            n=ARGS.sample,
            sigma_factor=ARGS.sigma_factor,
            shape_mode=ARGS.shape_mode,
            rng=ARGS.rng
        )
    else:
        strategy = RandomStrategy(
            ARGS.sample,
            *canvas.size(),
            alpha=ARGS.alpha,
            rng=ARGS.rng
        )
    return strategy


def show_function(canvas):
    return canvas.show_and_wait if ARGS.render_mode == 0 else canvas.show if ARGS.render_mode == 1 else lambda: None


def save_function(canvas):
    return (lambda output: canvas.save_in_size(output, ARGS.output_size)) if ARGS.save_actions else canvas.save


def save_every_action():
    return ARGS.output is not None and '%d' in ARGS.output


def save_final():
    return ARGS.output is not None and '%d' not in ARGS.output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, help='Number of triangles to draw', required=True)
    parser.add_argument('--input', type=str, help='Target image', required=True)
    parser.add_argument('--output', type=str, help='Output image')
    parser.add_argument('--render-mode', type=int,
                        help='Render mode: 0 - click, 1 - automatic, 2 - no render',
                        choices=[0, 1, 2], default=2)
    parser.add_argument('--alpha', type=float, help="Alpha value [0, 1]", default=0.5)
    parser.add_argument('--random', type=int, default=100)
    parser.add_argument('--sample', type=int, default=10)
    parser.add_argument('--step', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=4.64)
    parser.add_argument('--sigma-factor', type=float, default=0.03)
    parser.add_argument('--algorithm', type=str, choices=['random', 'simple', 'natural'],
                        default='natural')
    parser.add_argument('--optimizer', type=str,
                        choices=['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adadelta',
                                 'adam'], default='adam')
    parser.add_argument('--shape-mode', type=int,
                        help='Shape mode: 0 - all, 1 - triangle, 2 - rectangle, 3 - ellipse, '
                             '4 - quadrangle, 5 - brush', choices=[0, 1, 2, 3, 4, 5], default=0)
    parser.add_argument('--resize', type=int,
                        help='Size to which input will be scaled before drawing - the bigger the '
                             'longer it will take but the more details can be captured',
                        default=100)
    parser.add_argument('--output-size', type=int, help='Output image size', default=512)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('--save-actions', action='store_true', default=False, help="When not present, output-size "
                                                                                   "parameter will be ignored, and output image will be saved in size equal to resize parameter")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--metric', type=str, choices=['l1', 'l2'], default='l2')
    parser.add_argument('--scale-decay', type=float, default=0.00005)
    parser.add_argument('--background', type=str, help='Initial background color (hex value without #), if not passed, '
                                                       'will be the average target img color')
    ARGS = parser.parse_args()

    seed = ARGS.seed if ARGS.seed is not None else np.random.randint(0, 2 ** 32)
    rng = np.random.RandomState(seed=seed)
    log.info(f'Rng seed = {seed}')
    ARGS.rng = rng

    try:
        main()
    except Exception as e:
        logging.exception("")
