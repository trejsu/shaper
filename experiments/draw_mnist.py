import numpy as np
from keras.datasets import mnist

from es.drawer import Drawer

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--samples-start", type=int, default=0)
    parser.add_argument("--samples-end", type=int, default=1000)

    args = parser.parse_args()


    def data_mnist():
        (X, Y), (_, _) = mnist.load_data()
        X = X.reshape(X.shape[0], 28, 28, 1)
        X = X.astype('float32')
        X /= 255
        print("Loaded MNIST test data.")
        return X[args.samples_start:args.samples_end], \
               Y[args.samples_start:args.samples_end]


    X, Y = data_mnist()
    N = 100

    drawer = Drawer(
        alpha=0.7,
        background=None,
        save_all=True
    )

    X_drawings = drawer.draw(images=X, n=N)
    assert len(X_drawings) == N

    for n in range(1, N + 1):
        np.savez(
            args.output_path % n + f'-part-{args.part_index}',
            targets=X,
            drawings=X_drawings[n - 1],
            Y=Y
        )
