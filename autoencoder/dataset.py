import argparse

import numpy as np
from tqdm import tqdm

from shapes.canvas import Canvas
from shapes.shape import Triangle, Rectangle, Ellipse, Quadrangle, QuadrangleBrush, Curve

CHANNELS = 3


def main():
    train = np.empty((ARGS.num_train, ARGS.size, ARGS.size, CHANNELS))
    test = np.empty((ARGS.num_test, ARGS.size, ARGS.size, CHANNELS))
    canvas = Canvas.without_target(size=ARGS.size, background=(255, 255, 255), channels=CHANNELS)

    for i in tqdm(range(ARGS.num_train), desc=f'Generating train set'):
        shape = random_shape()
        canvas.add(shape)
        train[i] = canvas.img
        canvas.reset()

    for i in tqdm(range(ARGS.num_test), desc=f'Generating test set'):
        shape = random_shape()
        canvas.add(shape)
        test[i] = canvas.img
        canvas.reset()

    np.savez(ARGS.output_path, train=train, test=test)


def random_shape():
    shape = np.random.choice(ARGS.shapes)
    return {
        0: Triangle.random,
        1: Rectangle.random,
        2: Ellipse.random,
        3: Quadrangle.random,
        4: QuadrangleBrush.random,
        5: Curve.random,
    }[shape](w=ARGS.size, h=ARGS.size, alpha=0.8, rng=np.random.RandomState(), scale=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--shapes", type=str)
    parser.add_argument("--size", type=int)
    parser.add_argument("--num-train", type=int)
    parser.add_argument("--num-test", type=int)
    parser.add_argument("--output-path", type=str)
    ARGS = parser.parse_args()
    ARGS.shapes = np.fromstring(ARGS.shapes, dtype=int, sep=',')
    main()
