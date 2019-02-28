from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import spiral.utils as ut
from .base import Environment


class Triangles(Environment):
    action_sizes = {
        'color': [10, 10, 10],
        'alpha': [5],
        'p1': None,
        'p2': None,
        'p3': None
    }

    def __init__(self, args):
        super(Triangles, self).__init__(args)
        self._prepare_mnist()

    # todo: find out is it needed for conditional
    def get_random_target(self, num=1, squeeze=False):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def _prepare_mnist(self):
        ut.io.makedirs(self.args.data_dir)

        # ground truth MNIST data
        mnist_dir = self.args.data_dir / 'mnist'
        mnist = tf.contrib.learn.datasets.DATASETS['mnist'](str(mnist_dir))

        pkl_path = mnist_dir / 'mnist_dict.pkl'

        if pkl_path.exists():
            mnist_dict = ut.io.load_pickle(pkl_path)
        else:
            mnist_dict = defaultdict(lambda: defaultdict(list))
            for name in ['train', 'test', 'valid']:
                for num in self.args.mnist_nums:
                    filtered_data = \
                        mnist.train.images[mnist.train.labels == num]
                    filtered_data = \
                        np.reshape(filtered_data, [-1, 28, 28])

                    iterator = tqdm(filtered_data,
                                    desc="[{}] Processing {}".format(name, num))
                    for idx, image in enumerate(iterator):
                        # XXX: don't know which way would be the best
                        resized_image = ut.io.imresize(
                            image, [self.height, self.width],
                            interp='cubic')
                        mnist_dict[name][num].append(
                            np.expand_dims(resized_image, -1))
            ut.io.dump_pickle(pkl_path, mnist_dict)

        mnist_dict = mnist_dict['train' if self.args.train else 'test']

        data = []
        for num in self.args.mnist_nums:
            data.append(mnist_dict[int(num)])

        self.real_data = 255 - np.concatenate([d for d in data])
