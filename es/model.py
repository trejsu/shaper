from abc import abstractmethod

import keras.backend as K
import numpy as np
from keras.models import model_from_json


class Model(object):
    @abstractmethod
    def get_activations(self, X):
        raise NotImplementedError


class ModelA(Model):
    NUM_CLASSES = 10
    IMAGE_ROWS = 28
    IMAGE_COLS = 28
    NUM_CHANNELS = 1
    BATCH_SIZE = 100
    MODEL_PATH = '/Users/mchrusci/uj/blackbox-attacks/models/modelA'

    CONV1 = 0
    CONV2 = 2
    DENSE = 6

    def __init__(self, layer):
        assert layer in [self.CONV1, self.CONV2, self.DENSE]

        self.model = ModelA._load_model()
        self.layer = self.model.layers[layer].output

    @staticmethod
    def _load_model():
        print(f'Loading model from {ModelA.MODEL_PATH}')

        with open(ModelA.MODEL_PATH + '.json', 'r') as f:
            json_string = f.read()
            model = model_from_json(json_string)

        model.load_weights(ModelA.MODEL_PATH)
        return model

    def get_activations(self, X):
        assert 0 <= np.max(X) <= 1, f'np.max(X) = {np.max(X)}'

        if len(X.shape) == 3:
            X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])

        if X.shape[3] == 3:
            X = X[:, :, :, :1]

        N = X.shape[0]

        shape = self.layer.shape.as_list()
        shape[0] = N

        result = np.empty(shape)

        if self.BATCH_SIZE > N:
            batch_size = N
        else:
            batch_size = self.BATCH_SIZE

        num_batches = N // batch_size

        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size

            x = X[batch_start: batch_end]

            r = K.get_session().run(
                [self.layer],
                feed_dict={self.model.input: x, K.learning_phase(): 0}
            )[0]

            result[batch_start: batch_end] = r

        return result
