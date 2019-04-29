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
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--representation", action="store_true")
    parser.add_argument("--set", choices=["train", "test"], required=True)
    parser.add_argument("--shape-mode", choices=range(7), type=int, default=0)
    parser.add_argument("--save-targets", action="store_true")

    args = parser.parse_args()

    assert args.shape_mode or not args.representation, "Cannot use representation with shape mode = 0"


    def data_mnist():
        train, test = mnist.load_data()
        X, Y = train if args.set == "train" else test
        X = X.reshape(X.shape[0], 28, 28, 1)
        X = X.astype('float32')
        X /= 255
        print("Loaded MNIST test data.")
        return X[args.samples_start:args.samples_end], \
               Y[args.samples_start:args.samples_end]


    X, Y = data_mnist()

    drawer = Drawer(
        alpha=0.6,
        background=None,
        save_all=args.save_all,
        representation=args.representation,
        shape_mode=args.shape_mode,
        save_actions=args.representation
    )

    X_drawings = drawer.draw(images=X, n=args.n)

    if args.save_all:
        assert len(X_drawings) == args.n
    else:
        assert len(X_drawings) == X.shape[0]

    if args.save_all:
        for n in range(1, args.n + 1):
            if args.save_targets:
                np.savez(
                    args.output_path % n + f'-part-{args.part_index}',
                    targets=X,
                    drawings=X_drawings[n - 1],
                    Y=Y
                )
            else:
                np.savez(
                    args.output_path % n + f'-part-{args.part_index}',
                    drawings=X_drawings[n - 1],
                    Y=Y
                )

    else:
        if args.save_targets:
            np.savez(
                f'{args.output_path}-{args.n}-part-{args.part_index}',
                targets=X,
                drawings=X_drawings,
                Y=Y
            )
        else:
            np.savez(
                f'{args.output_path}-{args.n}-part-{args.part_index}',
                drawings=X_drawings,
                Y=Y
            )
