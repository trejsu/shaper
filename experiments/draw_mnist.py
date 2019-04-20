import numpy as np
from keras.datasets import mnist

from es.drawer import Drawer

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument("--coeffs", type=str, required=True)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--samples-start", type=int, default=0)
    parser.add_argument("--samples-end", type=int, default=1000)

    args = parser.parse_args()


    def data_mnist():
        (_, _), (X_test, Y_test) = mnist.load_data()
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_test = X_test.astype('float32')
        X_test /= 255
        print("Loaded MNIST test data.")
        return X_test[args.samples_start:args.samples_end], \
               Y_test[args.samples_start:args.samples_end]


    X, Y = data_mnist()
    N = 20

    drawer = Drawer(
        alpha=0.7,
        background='00',
        save_all=True,
        rewards=args.reward,
        rewards_coeffs=args.coeffs
    )

    num_samples = args.samples_end - args.samples_start

    X_drawings = drawer.draw(images=X, n=N)
    assert len(X_drawings) == N

    for n in range(1, N + 1):
        np.savez(
            args.output_path % n + f'-part-{args.part_index}',
            targets=X,
            drawings=X_drawings[n - 1],
            Y=Y
        )
