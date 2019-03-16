import math

import numpy as np
import tensorflow as tf
from capsNet import CapsNet
from capsnetConfig import cfg


class Classifier(object):
    def __init__(self, label):
        self.label = label

        model = CapsNet()
        graph = model.graph
        with graph.as_default():
            self.sess = tf.Session(graph=graph)
            saver = tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint(cfg.logdir))

        self.softmax_v = model.softmax_v
        self.X = model.X
        self.labels = model.labels

    def predict(self, X):
        x_size = X.shape[0]
        num_batches = math.ceil(x_size / cfg.batch_size)
        proba = np.empty((x_size,))

        for batch_idx in range(num_batches):
            labels = np.full((cfg.batch_size,), self.label)
            batch_start = batch_idx * cfg.batch_size
            batch_end = batch_start + cfg.batch_size
            batch = X[batch_start:batch_end]

            softmax = self.sess.run(self.softmax_v, {self.X: batch, self.labels: labels})
            proba[batch_start:batch_end] = softmax.reshape(cfg.batch_size, 10)[:, self.label]

        return proba
