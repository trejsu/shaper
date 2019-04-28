import numpy as np

from es.drawer import RepresentationDrawer

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", type=str, required=True)
    parser.add_argument("--shape", type=str, required=True)
    parser.add_argument("--size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--background", type=str)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    drawer = RepresentationDrawer(args.shape, args.size, args.channels, args.background)
    representation = np.load(args.represenation)
    imgs = drawer.draw(representation)
    np.savez(imgs, X=imgs)
