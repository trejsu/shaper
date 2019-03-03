from .shaper import MnistTriangles


def create_env(args):
    env = args.env.lower()
    if env == 'shaper':
        env = MnistTriangles(args)
    else:
        raise Exception(f"Unknown environment: {args.env}")
    return env
