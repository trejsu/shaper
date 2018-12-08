import argparse
import logging

from shaper.canvas import Canvas
from shaper.shape import Triangle

ARGS = None

log = logging.getLogger(__name__)


def main():
    canvas = Canvas(ARGS.input)

    for i in range(ARGS.n):
        log.info(f'Action {i}')
        t = Triangle.random(*canvas.size())
        canvas.add(t)
        canvas.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', type=int, help='Number of triangles to draw', required=True)
    parser.add_argument('--input', type=str, help='Target image', required=True)
    parser.add_argument('--output', type=str, help='Output image', required=True)
    ARGS = parser.parse_args()
    main()
