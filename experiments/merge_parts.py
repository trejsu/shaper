import os

import numpy as np

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-parts", type=int, required=True)
    parser.add_argument("--parts-path", type=str, required=True)
    parser.add_argument("--save-targets", action="store_true")
    parser.add_argument("--drawings-dim", nargs='+', type=int, required=True)

    args = parser.parse_args()
    args.drawings_dim = tuple(args.drawings_dim)

    part_size = args.num_samples // args.num_parts

    if '%d' in args.parts_path:
        for n in range(1, args.n + 1):
            if args.save_targets:
                targets = np.empty((args.num_samples, 28, 28, 1))
            drawings = np.empty((args.num_samples,) + args.drawings_dim)
            Y = np.empty((args.num_samples,))

            for part in range(args.num_parts):
                part_start = part * part_size
                part_end = part_start + part_size

                part_path = (args.parts_path % n) + f'-part-{part}.npz'
                part_data = np.load(part_path)
                if args.save_targets:
                    targets[part_start:part_end] = part_data['targets']
                drawings[part_start:part_end] = part_data['drawings']
                Y[part_start:part_end] = part_data['Y']

            if args.save_targets:
                np.savez(args.parts_path % n, targets=targets, drawings=drawings, Y=Y)
            else:
                np.savez(args.parts_path % n, drawings=drawings, Y=Y)

            for part in range(args.num_parts):
                part_path = (args.parts_path % n) + f'-part-{part}.npz'
                os.remove(part_path)
    else:
        if args.save_targets:
            targets = np.empty((args.num_samples, 28, 28, 1))
        drawings = np.empty((args.num_samples,) + args.drawings_dim)
        Y = np.empty((args.num_samples,))

        for part in range(args.num_parts):
            part_start = part * part_size
            part_end = part_start + part_size

            part_path = f'{args.parts_path}-part-{part}.npz'
            part_data = np.load(part_path)
            if args.save_targets:
                targets[part_start:part_end] = part_data['targets']
            drawings[part_start:part_end] = part_data['drawings']
            Y[part_start:part_end] = part_data['Y']

        if args.save_targets:
            np.savez(args.parts_path, targets=targets, drawings=drawings, Y=Y)
        else:
            np.savez(args.parts_path, drawings=drawings, Y=Y)

        for part in range(args.num_parts):
            part_path = f'{args.parts_path}-part-{part}.npz'
            os.remove(part_path)
