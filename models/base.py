import math
import numpy as np

from models import evaluation


class EcgModel:
    def name(self):
        return type(self).__name__.lower()

    def restore(self):
        raise NotImplementedError()

    def fit(self, x, y, validation=None):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        evaluation.print_validation_info(y_true, y_pred)

    @staticmethod
    def get_class_weights(y, mu=0.8):
        """
        :param y: labels
        :param mu: parameter to tune
        :return: class weights dictionary
        """
        train_categories_dist = dict()
        labels = np.unique(y)
        for label in labels:
            train_occurancies = sum([1 if label == y else 0 for y in Y])
            train_categories_dist[label] = train_occurancies

        total = sum(train_categories_dist.values())
        keys = train_categories_dist.keys()
        class_weight = dict()

        for key in keys:
            score = math.log(mu * total / float(train_categories_dist[key]))
            class_weight[key] = score if score > 1.0 else 1.0

        return class_weight
