import numpy as np

from es.drawer import Drawer

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-path", type=str, required=True)

    args = parser.parse_args()

    with np.load(args.samples_path) as data:
        X = data['X']
        Y = data['Y']

    N = 100

    drawer = Drawer(
        alpha=0.6,
        background=None,
        save_all=True
    )

    X_drawings = drawer.draw(images=X, n=N)
    assert len(X_drawings) == N

    base_path = args.samples_path.split('.npz')[0]

    for n in range(1, N + 1):
        np.savez(
            f'{base_path}-redrawned-{n}',
            targets=X,
            drawings=X_drawings[n - 1],
            Y=Y
        )
