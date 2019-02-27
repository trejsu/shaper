from .shaper import Triangles


def create_env(args):
    env = args.env.lower()
    if env == 'shaper':
        env = Triangles(args)
    else:
        raise Exception("Unkown environment: {}".format(args.env))
    return env
