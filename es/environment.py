import logging
from abc import abstractmethod

import numpy as np

from es.classifier import Classifier
from shapes.util import l2_full, l1_full, update_l1, update_l2

log = logging.getLogger(__name__)


class Environment(object):
    def __init__(self, canvas, save_actions, num_shapes):
        self.canvas = canvas
        self.save_actions = save_actions
        self.current_shape_num = 0
        self.shapes = [None] * num_shapes
        self.prev_img = None

    @abstractmethod
    def init(self):
        raise NotImplementedError

    def observation_shape(self):
        return self.canvas.size()

    @abstractmethod
    def evaluate(self, shape):
        raise NotImplementedError

    @abstractmethod
    def evaluate_batch(self, shapes):
        raise NotImplementedError

    @abstractmethod
    def step(self, shape):
        raise NotImplementedError

    def save_in_size(self, output, size):
        if self.save_actions:
            log.debug(f'Saving image under {output} in size = {size}')

            canvas = self.canvas.clear_and_resize(size=size)
            w, h = canvas.size()

            for s in self.shapes:
                if s is None:
                    break
                else:
                    cls, normalized_params = s

                shape = cls.from_normalized_params(w, h, *normalized_params)
                canvas.add(shape=shape)

            canvas.save(output=output)
        else:
            raise Exception("Cannot save in size, save_actions set to False")

    @abstractmethod
    def _undo(self):
        raise NotImplementedError

    def _remove_last_shape(self):
        self.current_shape_num -= 1

    def _save_shape(self, shape):
        self.shapes[self.current_shape_num] = (
            shape.__class__,
            shape.normalized_params(*self.observation_shape())
        )
        self.current_shape_num += 1


class DistanceEnvironment(Environment):

    def __init__(self, canvas, save_actions, num_shapes, metric):
        super().__init__(canvas, save_actions, num_shapes)
        self.metric = metric
        self.distance = (l1_full if metric == 'l1' else l2_full)(
            target=self.canvas.target,
            x=self.canvas.img
        )
        self.update_distance = update_l1 if metric == 'l1' else update_l2
        self.prev_distance = None

    def evaluate(self, shape):
        score = self.step(shape)
        self._undo()
        return score

    def evaluate_batch(self, shapes):
        return [self.evaluate(shape) for shape in shapes]

    def step(self, shape):
        self.prev_img = self.canvas.img.copy()
        self.prev_distance = self.distance.copy()

        bounds = self.canvas.add(shape)

        self.update_distance(
            distance=self.distance,
            bounds=bounds,
            img=self.canvas.img,
            target=self.canvas.target
        )

        if self.save_actions:
            self._save_shape(shape)

        return self._score()

    def _score(self):
        return np.average(self.distance)

    def _undo(self):
        self.canvas.img = self.prev_img
        self.distance = self.prev_distance
        if self.save_actions:
            self._remove_last_shape()

    def init(self):
        return self._score()


class ClassificationEnvironment(Environment):

    def __init__(self, canvas, save_actions, num_shapes, label):
        super().__init__(canvas, save_actions, num_shapes)
        self.classifier = Classifier(label)

    def evaluate(self, shape):
        raise NotImplementedError

    def evaluate_batch(self, shapes):
        X = np.empty((len(shapes), 28, 28, 1))

        for i, shape in enumerate(shapes):
            self.step(shape)
            x = self.canvas.img
            # self.canvas.save(f'/tmp/{shape}.jpg')
            self._undo()
            X[i] = x[:, :, :1] / 255

        # to minimalize
        probs = self.classifier.predict(X)
        # print(f'probs = {probs}')
        return 1 - probs

    def step(self, shape):
        self.prev_img = self.canvas.img.copy()
        self.canvas.add(shape)

    def _undo(self):
        self.canvas.img = self.prev_img
        if self.save_actions:
            self._remove_last_shape()

    def init(self):
        raise NotImplementedError


class MixedEnvironment(Environment):

    def __init__(self, canvas, save_actions, num_shapes, metric, label):
        super().__init__(canvas, save_actions, num_shapes)
        self.metric = metric
        self.distance = (l1_full if metric == 'l1' else l2_full)(
            target=self.canvas.target,
            x=self.canvas.img
        )
        self.update_distance = update_l1 if metric == 'l1' else update_l2
        self.prev_distance = None
        self.classifier = Classifier(label)

    def init(self):
        raise NotImplementedError

    def evaluate(self, shape):
        raise NotImplementedError

    def evaluate_batch(self, shapes):
        X = np.empty((len(shapes), 28, 28, 1))
        distances = np.empty((len(shapes, )))

        for i, shape in enumerate(shapes):
            distances[i] = self.step(shape)
            x = self.canvas.img
            self._undo()
            X[i] = x[:, :, :1] / 255

        probs = self.classifier.predict(X)
        return distances / probs

    def step(self, shape):
        self.prev_img = self.canvas.img.copy()
        self.prev_distance = self.distance.copy()

        bounds = self.canvas.add(shape)

        self.update_distance(
            distance=self.distance,
            bounds=bounds,
            img=self.canvas.img,
            target=self.canvas.target
        )

        if self.save_actions:
            self._save_shape(shape)

        return self._score()

    def _undo(self):
        self.canvas.img = self.prev_img
        self.distance = self.prev_distance
        if self.save_actions:
            self._remove_last_shape()

    def _score(self):
        return np.average(self.distance)
