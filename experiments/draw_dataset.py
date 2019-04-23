import numpy as np

from es.drawer import Drawer


def load_data():
    with np.load(args.samples_path) as data:
        X = data['X']
        Y = data['Y']

    if args.multiplication != 1:
        X = np.repeat(X, args.multiplication, axis=0)
        Y = np.repeat(Y, args.multiplication, axis=0)
        np.random.seed(666)
        np.random.shuffle(X)
        np.random.seed(666)
        np.random.shuffle(Y)

    return X, Y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--save-all", type=bool, required=True)
    parser.add_argument("--multiplication", type=int, default=1)

    args = parser.parse_args()

    X, Y = load_data()

    drawer = Drawer(
        alpha=0.6,
        background=None,
        save_all=args.save_all
    )

    X_drawings = drawer.draw(images=X, n=args.n)

    if args.save_all:
        assert len(X_drawings) == args.n
    else:
        assert len(X_drawings) == X.shape[0]

    base_path = args.samples_path.split('.npz')[0]

    if args.save_all:
        for n in range(1, args.n + 1):
            np.savez(
                f'{base_path}-redrawned-{n}',
                targets=X,
                drawings=X_drawings[n - 1],
                Y=Y
            )
    else:
        np.savez(f'{base_path}-redrawn', targets=X, drawings=X_drawings, Y=Y)
