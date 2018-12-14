import argparse
import logging
import time

from shaper.canvas import Canvas
from shaper.shape import Shape

ARGS = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    canvas = Canvas(ARGS.input)
    show = show_function(canvas)
    best_score = canvas.init()
    log.info(f'Initial score: {best_score}')

    global_start = time.time()
    global_tries = 0

    for i in range(1, ARGS.n + 1):
        start = time.time()
        tries = 0
        shape = Shape.random(*canvas.size(), alpha=ARGS.alpha)
        tries += 1
        score = canvas.add(shape)
        log.debug(f'Action {i}, try {tries}, best score: {best_score}, score: {score}')
        show()

        while score > best_score:
            canvas.undo()
            shape = Shape.random(*canvas.size(), alpha=ARGS.alpha)
            tries += 1
            score = canvas.add(shape)
            log.debug(f'Action {i}, try {tries}, best score: {best_score}, score: {score}')
            show()

        # show()

        elapsed = time.time() - start
        log.info(f'Action {i}, shapes drawned {tries}, time {elapsed:.2f} s, '
                 f'({tries / elapsed:.1f} shapes/s), new score: {score:.4f}, '
                 f'score delta {best_score - score:.4f}')
        best_score = score
        global_tries += tries

    global_elapsed = time.time() - global_start
    log.info(f'Total shapes drawned {global_tries}, time {global_elapsed:.2f} s, '
             f'({global_tries / global_elapsed:.1f} shapes/s)')

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
