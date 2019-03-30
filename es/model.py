import math
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from capsNet import CapsNet
from capsnetConfig import cfg
from dcgan import FLAGS
from dcganModel import DCGAN


class Model(object):
    def __init__(self, label):
        self.label = label

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError


class Classifier(Model):
    def __init__(self, label):
        super().__init__(label)

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
        probs = np.empty((x_size,))

        for batch_idx in range(num_batches):
            labels = np.full((cfg.batch_size,), self.label)
            batch_start = batch_idx * cfg.batch_size
            batch_end = batch_start + cfg.batch_size
            batch = X[batch_start:batch_end]

            softmax = self.sess.run(self.softmax_v, {self.X: batch, self.labels: labels})
            probs[batch_start:batch_end] = softmax.reshape(cfg.batch_size, 10)[:, self.label]

        return probs


class Discriminator(Model):
    def __init__(self, label):
        super().__init__(label)

        run_config = tf.ConfigProto(intra_op_parallelism_threads=FLAGS.cpu,
                                    inter_op_parallelism_threads=FLAGS.cpu)
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        dcgan = DCGAN(
            self.sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.dcgan_batch_size,
            sample_num=FLAGS.dcgan_batch_size,
            y_dim=10,
            z_dim=FLAGS.generate_test_images,
            dataset_name=FLAGS.dcgan_dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
            data_dir=FLAGS.data_dir
        )

        dcgan.load(FLAGS.checkpoint_dir)

        self.real_prob = dcgan.D_
        self.logits = dcgan.D_logits_
        self.X = dcgan.G
        self.Y = dcgan.y

    def predict(self, X):
        x_size = X.shape[0]
        num_batches = math.ceil(x_size / FLAGS.dcgan_batch_size)
        result = np.empty((x_size,))

        for batch_idx in range(num_batches):
            labels = np.eye(10)[np.full((FLAGS.dcgan_batch_size,), self.label, dtype=np.int)]
            batch_start = batch_idx * FLAGS.dcgan_batch_size
            batch_end = batch_start + FLAGS.dcgan_batch_size
            batch = X[batch_start:batch_end]

            real_prob, logits = self.sess.run([self.real_prob, self.logits],
                                              {self.X: batch, self.Y: labels})
            # print('real_prob', real_prob)
            # print('logits', logits)
            result[batch_start:batch_end] = real_prob.reshape(10, )

        return result
