from .base import Environment


class Triangles(Environment):
    action_sizes = {
        'color': [10, 10, 10],
        'alpha': [5],
        'p1': None,
        'p2': None,
        'p3': None
    }

    def get_random_target(self, num=1, squeeze=False):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass
