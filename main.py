import argparse
import logging
import time

from shaper.canvas import Canvas
from shaper.strategy import SimpleEvolutionStrategy

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# todo: move to some config
NUM_SOLUTIONS = 100
NUM_IMPROVEMENTS = 10


def main():
    canvas = Canvas(ARGS.input)
    show = show_function(canvas)
    score = canvas.init()
    log.info(f'Initial score: {score}')

    start = time.time()

    for i in range(1, ARGS.n + 1):
        strategy = SimpleEvolutionStrategy(NUM_SOLUTIONS, *canvas.size(), alpha=ARGS.alpha)
        best_score = 9223372036854775807
        best_shape = None

        for j in range(1, NUM_IMPROVEMENTS + 1):
            shapes = strategy.ask()
            scores = [canvas.evaluate(shape) for shape in shapes]
            strategy.tell(scores)
            shape, score = strategy.result()

            if score < best_score:
                best_score = score
                best_shape = shape

        log.info(f'Best shape: {best_shape}')
        score = canvas.add(best_shape)
        log.info(f'Action {i}, new score: {score:.4f}')
        show()

    elapsed = time.time() - start
    shapes_drawn = ARGS.n * NUM_IMPROVEMENTS * NUM_SOLUTIONS
    log.info(f'Total shapes drawn {shapes_drawn}, time {elapsed:.2f} s, '
             f'({shapes_drawn / elapsed:.1f} shapes/s)')

    if ARGS.output is not None:
        canvas.save(ARGS.output)


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
    ARGS = parser.parse_args()
    main()
