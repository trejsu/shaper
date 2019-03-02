import numpy as np
import tensorflow as tf

from shapes.canvas import Canvas
from shapes.shape import Triangle
from spiral.envs.base import Environment
from spiral.envs.utils import uniform_locations, l2_dist


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
        self.colors = np.array(self.colors)
        self.alphas = np.linspace(0.2, 1, 5)
        self.p1s = uniform_locations(self.screen_size, self.location_size)
        self.p2s = uniform_locations(self.screen_size, self.location_size)
        self.p3s = uniform_locations(self.screen_size, self.location_size)

        self.canvas = None
        self.step_num = None

    @property
    def state(self):
        return self.canvas.img

    @property
    def target(self):
        return self.canvas.target

    def step(self, ac):
        self._draw(ac)
        self.step_num += 1

        terminal_action = (self.step_num == self.episode_length)
        if terminal_action:
            reward = 1
            # todo: partial l2
            l2 = l2_dist(self.state, self.target)
            punishment = l2 / np.prod(self.observation_shape)
            reward -= punishment
        else:
            reward = 0
        return self.state, reward, terminal_action, {}

    def reset(self):
        target = self.get_random_target(num=1, squeeze=True)
        self.canvas = Canvas(target=target, size=self.screen_size, background=ut.color.WHITE)
        self.step_num = 0
        # todo: find out how z is used (refer to paper)
        self.z = None
        return self.state, self.target, self.z

    def get_random_target(self, num=1, squeeze=False):
        random_idxes = np.random.choice(self.real_data.shape[0], num, replace=False)
        random_image = self.real_data[random_idxes]
        if squeeze:
            random_image = np.squeeze(random_image, 0)
        return random_image

    def get_action_desc(self, ac):
        desc = []
        for name in self.action_sizes:
            named_ac = ac[self.ac_idx[name]]
            actual_ac = getattr(self, name + "s")[named_ac]
            desc.append(f"{name}: {actual_ac} ({named_ac})")
        return "\n".join(desc)

    def _prepare_mnist(self):
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        data = x_train if self.args.train else x_test
        self.real_data = 255 - data

    def save_image(self, path):
        self.canvas.save(path)

    def _draw(self, ac):
        color = self.colors[0]
        alpha = self.alphas[-1]
        p1_x, p1_y = self.p1s[0]
        p2_x, p2_y = self.p2s[0]
        p3_x, p3_y = self.p3s[0]

        for name in self.action_sizes:
            named_action = ac[self.ac_idx[name]]
            value = getattr(self, name + "s")[named_action]

            if name == 'color':
                color = value
            elif name == 'alpha':
                alpha = value
            elif name == 'p1':
                p1_x, p1_y = value
            elif name == 'p2':
                p2_x, p2_y = value
            elif name == 'p3':
                p3_x, p3_y = value

        t = Triangle.from_params(p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, alpha)
        self.canvas.add(shape=t, color=color)


if __name__ == '__main__':
    import spiral.utils as ut
    from spiral.config import get_args

    args = get_args()
    ut.train.set_global_seed(args.seed)

    env = MnistTriangles(args)

    for ep_idx in range(10):
        step = 0
        env.reset()

        while True:
            action = env.random_action()
            print(f"[Step {step}] ac: {env.get_action_desc(action)}")
            state, reward, terminal, info = env.step(action)
            env.save_image(f"mnist{ep_idx}_{step}.png")

            if terminal:
                print(f"Ep #{ep_idx} finished ==> Reward: {reward}")
                break

            step += 1
