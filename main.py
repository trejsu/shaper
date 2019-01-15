import argparse
import logging
import time

from shaper.canvas import Canvas
from shaper.optimizer import GradientDescent, Adam, Momentum, Nesterov, Adadelta, Adagrad, RMSProp
from shaper.strategy import RandomStrategy, EvolutionStrategy, SimpleEvolutionStrategy
from shaper.util import print_times

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    canvas, show = init()
    start = time.time()

    for i in range(1, ARGS.n + 1):
        best_score, best_shape = find_best_random_shape(canvas)

        strategy = pick_strategy(best_shape, canvas)

        for j in range(1, ARGS.step + 1):
            score, shape = find_best_shape(canvas, strategy)

            if score < best_score:
                best_score = score
                best_shape = shape

        score = canvas.add(best_shape)
        log.info(f'Action {i}, new score: {score:.4f}')
        show()

    elapsed = time.time() - start
    shapes_drawn = ARGS.n * (ARGS.step * ARGS.sample + ARGS.random)
    log.info(f'Total shapes drawn {shapes_drawn}, time {elapsed:.2f} s, '
             f'({shapes_drawn / elapsed:.1f} shapes/s)')

    if ARGS.output is not None:
        canvas.save(ARGS.output)

    if ARGS.time:
        print_times()


def find_best_shape(canvas, strategy):
    shapes = strategy.ask()
    scores = [canvas.evaluate(shape) for shape in shapes]
    strategy.tell(scores)
    shape, score = strategy.result()
    return score, shape


def init():
    canvas = Canvas(
        target=ARGS.input,
        size=ARGS.resize,
        output_size=ARGS.output_size,
        num_shapes=ARGS.n
    )
    show = show_function(canvas)
    score = canvas.init()
    log.info(f'Initial score: {score}')
    return canvas, show


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
            shape_mode=ARGS.shape_mode
        )
    elif ARGS.algorithm == 'simple':
        strategy = SimpleEvolutionStrategy(
            best_shape,
            *canvas.size(),
            alpha=ARGS.alpha,
            n=ARGS.sample,
            sigma_factor=ARGS.sigma_factor,
            shape_mode=ARGS.shape_mode
        )
    else:
        strategy = RandomStrategy(
            ARGS.sample,
            *canvas.size(),
            alpha=ARGS.alpha
        )
    return strategy


def find_best_random_shape(canvas):
    random = RandomStrategy(
        ARGS.random,
        *canvas.size(),
        alpha=ARGS.alpha,
        shape_mode=ARGS.shape_mode
    )
    shapes = random.ask()
    scores = [canvas.evaluate(shape) for shape in shapes]
    random.tell(scores)
    best_shape, best_score = random.result()
    return best_score, best_shape


def show_function(canvas):
    return canvas.show_and_wait if ARGS.render_mode == 0 else canvas.show if ARGS.render_mode == 1 else lambda: None


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
                             '4 - curve', choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('--resize', type=int,
                        help='Size to which input will be scaled before drawing - the bigger the '
                             'longer it will take but the more details can be captured',
                        default=100)
    parser.add_argument('--output-size', type=int, help='Output image size', default=512)
    parser.add_argument('--time', action='store_true', default=False)
    ARGS = parser.parse_args()
    main()
