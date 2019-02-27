# Code based on https://github.com/carpedm20/SPIRAL-tensorflow/blob/master/envs/base.py
import numpy as np


class Environment(object):

    def __init__(self, args):
        self.args = args

        # terminal
        self.episode_length = args.episode_length

        # screen
        self.screen_size = args.screen_size
        self.height, self.width = self.screen_size, self.screen_size
        self.observation_shape = [self.height, self.width, args.color_channel]

        # location
        self.location_size = args.location_size
        self.location_shape = [self.location_size, self.location_size]

        for name, value in self.action_sizes.items():
            if value is None:
                self.action_sizes[name] = self.location_shape

        self.acs = list(self.action_sizes.keys())
        self.ac_idx = {
            ac: idx for idx, ac in enumerate(self.acs)
        }

    def random_action(self):
        action = []
        for ac in self.acs:
            size = self.action_sizes[ac]
            sample = np.random.randint(np.prod(size))
            action.append(sample)
        return action

    @property
    def initial_action(self):
        return [-1] * len(self.acs)

    def norm(self, img):
        return (np.array(img) - 127.5) / 127.5

    def denorm(self, img):
        return img * 127.5 + 127.5

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def get_random_target(self, num=1, squeeze=False):
        raise NotImplementedError
