import numpy as np

from es.drawer import Drawer


def load_data():
    with np.load(args.samples_path) as data:
        X = data['X']
        Y = data['Y']

    Y = np.argmax(Y, axis=1).reshape(-1, )

    if args.multiplication != 1:
        X = np.repeat(X, args.multiplication, axis=0)
        Y = np.repeat(Y, args.multiplication, axis=0)
        np.random.seed(666)
        np.random.shuffle(X)
        np.random.seed(666)
        np.random.shuffle(Y)

    assert X.shape[1] == 28 and X.shape[2] == 28 and X.shape[3] == 1, f'X.shape = {X.shape}'
    assert len(Y.shape) == 1, f'Y.shape = {Y.shape}'

    return X, Y


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--save-all", action="store_true")
    parser.add_argument("--multiplication", type=int, default=1)
    parser.add_argument("--shape-mode", choices=range(7), type=int, default=0)
    parser.add_argument("--save-targets", action="store_true")
    parser.add_argument("--representation", action="store_true")

    args = parser.parse_args()

    assert args.shape_mode or not args.representation, "Cannot use representation with shape mode = 0"

    X, Y = load_data()
    print(f'Y.shape = {Y.shape}')

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

    base_path = args.samples_path.split('.npz')[0]

    if args.save_all:
        for n in range(1, args.n + 1):
            path = f'{base_path}-redrawn-{n}'
            drawings = X_drawings[n - 1]
            if args.save_targets:
                np.savez(path, targets=X, drawings=drawings, Y=Y)
            else:
                np.savez(path, drawings=drawings, Y=Y)
    else:
        path = f'{base_path}-redrawn-{args.n}'
        if args.save_targets:
            np.savez(path, targets=X, drawings=X_drawings, Y=Y)
        else:
            np.savez(path, drawings=X_drawings, Y=Y)
