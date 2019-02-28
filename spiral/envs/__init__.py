from .shaper import MnistTriangles


def create_env(args):
    env = args.env.lower()
    if env == 'shaper':
        env = MnistTriangles(args)
    else:
        raise Exception("Unkown environment: {}".format(args.env))
    return env
