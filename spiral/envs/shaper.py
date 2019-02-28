from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import spiral.envs.utils as utils
import spiral.utils as ut
from shaper.canvas import Canvas
from .base import Environment


class MnistTriangles(Environment):
    action_sizes = {
        'color': [2],
        'alpha': [5],
        'p1': None,
        'p2': None,
        'p3': None
    }

    def __init__(self, args):
        super(MnistTriangles, self).__init__(args)
        self._prepare_mnist()

        self.colors = [
            ut.color.BLACK,
            ut.color.WHITE
        ]
        self.colors = np.array(self.colors) / 255.
        self.alphas = np.linspace(0.2, 1, 5)
        self.p1s = utils.uniform_locations(self.screen_size, self.location_size, object_radius=0)
        self.p2s = utils.uniform_locations(self.screen_size, self.location_size, object_radius=0)
        self.p3s = utils.uniform_locations(self.screen_size, self.location_size, object_radius=0)

        self.target = None
        self.canvas = None
        self.step = None

    @property
    def state(self):
        return self.canvas.img

    def step(self, action):
        pass

    def reset(self):
        self.target = self.get_random_target(num=1, squeeze=True)
        self.canvas = Canvas(target=self.target, size=self.screen_size, background=ut.color.WHITE)
        self.step = 0
        # todo: find out how z is used (refer to paper)
        self.z = None
        return self.state, self.target, self.z

    def get_random_target(self, num=1, squeeze=False):
        random_idxes = np.random.choice(self.real_data.shape[0], num, replace=False)
        random_image = self.real_data[random_idxes]
        if squeeze:
            random_image = np.squeeze(random_image, 0)
        return random_image

    def _prepare_mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0


        ut.io.makedirs(self.args.data_dir)

        mnist_dir = self.args.data_dir / 'mnist'
        mnist = tf.contrib.learn.datasets.DATASETS['mnist'](str(mnist_dir))

        pkl_path = mnist_dir / 'mnist_dict.pkl'

        if pkl_path.exists():
            mnist_dict = ut.io.load_pickle(pkl_path)
        else:
            mnist_dict = defaultdict(lambda: defaultdict(list))
            for name in ['train', 'test', 'valid']:
                for num in self.args.mnist_nums:
                    filtered_data = mnist.train.images[mnist.train.labels == num]
                    filtered_data = np.reshape(filtered_data, [-1, 28, 28])

                    iterator = tqdm(filtered_data, desc="[{}] Processing {}".format(name, num))
                    for idx, image in enumerate(iterator):
                        # XXX: don't know which way would be the best
                        resized_image = ut.io.imresize(image, [self.height, self.width], interp='cubic')
                        mnist_dict[name][num].append(np.expand_dims(resized_image, -1))
            ut.io.dump_pickle(pkl_path, mnist_dict)

        mnist_dict = mnist_dict['train' if self.args.train else 'test']

        data = []
        for num in self.args.mnist_nums:
            data.append(mnist_dict[int(num)])

        self.real_data = 255 - np.concatenate([d for d in data])
