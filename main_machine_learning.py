import random

import numpy as np

from utils import async
from utils import logger

logger.log_to_files('ml')

# seed = int(random.random() * 1e6)
from sklearn.ensemble import RandomForestClassifier

seed = 42
random.seed(seed)
np.random.seed(seed)
print("Seed =", seed)

import loader
import preprocessing
import keras_helper as helper

from scipy import stats

from sklearn.metrics import confusion_matrix, accuracy_score

from models import *

from feature_extractor import *

import numpy as np

FREQUENCY = 300  # 300 points per second
WINDOW_SIZE = int(0.8 * FREQUENCY)
STEP = int(0.2 * FREQUENCY)
FADEIN = int(1.5 * FREQUENCY)
NB_PERIODS = 3


def features_for_row(row):
    features = []

    features += wavelet_coefficients(row)

    pqrsts = extract_pqrst(row)

    features.append(len(pqrsts) * 1.0 * FREQUENCY / len(row))

    if len(pqrsts) == 0:
        return features + [0 for x in range(5 + 7 * 12)]

    p = [x[0] for x in pqrsts]
    q = [x[1] for x in pqrsts]
    r = [x[2] for x in pqrsts]
    s = [x[3] for x in pqrsts]
    t = [x[4] for x in pqrsts]

    rrs = np.diff(r)

    if len(rrs) > 0:
        features += [
            np.amin(rrs),
            np.amax(rrs),
            np.mean(rrs),
            np.std(rrs),
            sum([1 for x in r if x < 0])
        ]
    else:
        features += [0 for x in range(5)]

    pqrsts = pqrsts[:min(7, len(pqrsts))]
    row = low_pass_filtering(row)
    row = high_pass_filtering(row)
    row = derivative_filter(row)
    row = squaring(row)
    row = moving_window_integration(row)
    for i in range(len(pqrsts)):
        pq = row[p[i]:q[i]]
        st = row[s[i]:t[i]]
        pt = row[p[i]:t[i]]
        pmax = np.amax(pq)
        tmax = np.amax(st)

        features += [
            # features for PQ interval
            pmax,
            pmax / row[r[i]],
            np.mean(pq),
            np.std(pq),
            stats.mode(pq).mode[0],

            # feature for ST interval
            tmax,
            tmax / row[r[i]],
            np.mean(st),
            np.std(st),
            stats.mode(st).mode[0],

            # features for whole PQRST interval
            stats.skew(pt),
            stats.kurtosis(pt)
        ]

    for i in range(7 - len(pqrsts)):
        features += [0 for x in range(12)]

    return features


(X, Y) = loader.load_all_data()
X = preprocessing.normalize(X)
(Y, mapping) = preprocessing.format_labels(Y)
print('Input length', len(X))
print('Categories mapping', mapping)

subX = X
subY = Y

print("Features extraction started")
subX = async.apply_async(subX, features_for_row)
print("Features extraction finished")
subY = subY

from collections import Counter

print("Distribution of categories before balancing")
counter = Counter(subY)
for key in counter.keys():
    print(key, counter[key])

Xt, Xv, Yt, Yv = helper.train_test_split(subX, subY, 0.33)

model = RandomForestClassifier(n_estimators=20, n_jobs=mp.cpu_count())
model.fit(Xt, Yt)
print(model.feature_importances_)

Ypredicted = model.predict(Xv)

accuracy = accuracy_score(Yv, Ypredicted)
print(accuracy)
matrix = confusion_matrix(Yv, Ypredicted)
print(matrix)
