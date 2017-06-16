from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from models.base import EcgModel


class RandomForestEcgModel(EcgModel):
    rf = None
    model_file = "model.pkl"

    def restore(self):
        self.rf = joblib.load(self.model_file)

    def fit(self, x, y, validation=None):
        self.rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, n_jobs=-1)
        self.rf.fit(x, y)
        joblib.dump(self.rf, self.model_file)

    def predict(self, x):
        return self.rf.predict(x)

    def show_feature_importances(self, features_names=None):
        import matplotlib.pyplot as plt
        import numpy as np

        importances = self.rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.rf.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        if features_names is None:
            labels = indices
        else:
            labels = [features_names[i] for i in indices]

        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(len(importances)), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(len(importances)), labels, rotation='vertical', fontsize=10)
        plt.subplots_adjust(bottom=0.3)
        plt.xlim([-1, len(importances)])
        plt.show()
