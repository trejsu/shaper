# Code based on https://github.com/carpedm20/SPIRAL-tensorflow/blob/master/replay.py
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args, observation_shape):
        self.buffer_size = args.buffer_batch_num * args.disc_batch_size
        self.batch_size = args.disc_batch_size

        self.rng = np.random.RandomState(args.seed)

        self.idx = 0
        replay_shape = [self.buffer_size] + observation_shape
        self.data = np.zeros(replay_shape, dtype=np.uint8)

        self.most_recent = None

    def push(self, batches):
        batch_size = len(batches)
        if self.idx + batch_size > self.buffer_size:
            self.data[:-batch_size] = self.data[batch_size:]
            self.data[-batch_size:] = batches
        else:
            self.data[self.idx:self.idx + batch_size] = batches
            self.idx += int(batch_size)

    def sample(self, n):
        while self.idx < n:
            pass
        random_idx = self.rng.choice(
            self.idx, self.batch_size)
        return self.data[random_idx].astype(np.float32)
