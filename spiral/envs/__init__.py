from .shaper import Shaper


def create_env(args):
    env = args.env.lower()
    if env == 'shaper':
        env = Shaper(args)
    else:
        raise Exception("Unkown environment: {}".format(args.env))
    return env
