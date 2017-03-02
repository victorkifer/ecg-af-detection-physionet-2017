import random
import numpy as np

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
import math

FREQUENCY = 300  # 300 points per second
WINDOW_SIZE = int(0.8 * FREQUENCY)
STEP = int(0.2 * FREQUENCY)
FADEIN = int(1.5 * FREQUENCY)
NB_PERIODS = 3


def range_presentation(row, value, percent):
    amax = math.fabs(np.amax(row))
    min = value - percent * amax
    max = value + percent * amax
    total = 1.0 * len(row)
    return sum([1 if min < x < max else 0 for x in row]) / total


def relative_mean(row, trim, mean):
    if mean == 0:
        mean = 1.0
    return np.mean(stats.trimboth(row, trim)) / mean


def features_for_period(row, start, end):
    subrow = row[start:end]
    mode = stats.mode(subrow).mode[0]
    if math.isnan(mode):
        raise Exception('No mode')
    mean = np.mean(subrow)
    return [
        np.amin(subrow),
        np.amax(subrow),
        mean,
        np.std(subrow),
        relative_mean(subrow, 0.1, mean),
        relative_mean(subrow, 0.2, mean),
        relative_mean(subrow, 0.3, mean),
        mode,
        range_presentation(subrow, mode, 0.05),
        range_presentation(subrow, mode, 0.1),
        range_presentation(subrow, mode, 0.2),
        range_presentation(subrow, mode, 0.4),
        stats.skew(subrow),
        stats.kurtosis(subrow)
    ]


def features_for_row(row):
    start = FADEIN
    times = (NB_PERIODS - 1) * WINDOW_SIZE // STEP
    features = []
    features_per_period = None
    for i in range(times):
        features += features_for_period(row, start, start + WINDOW_SIZE)
        if features_per_period is None:
            features_per_period = len(features)
        start += STEP

    mins = np.array(features[0::features_per_period])
    maxs = np.array(features[1::features_per_period])
    means = np.array(features[2::features_per_period])
    stds = np.array(features[3::features_per_period])

    for c in (mins, maxs, means, stds):
        features.append(np.amin(c))
        features.append(np.amax(c))
        features.append(np.mean(c))
        features.append(np.std(c))

    r_peaks = np.array(get_r_peaks_frequencies(row))
    if len(r_peaks) > 0:
        features.append(np.amin(r_peaks))
        features.append(np.amax(r_peaks))
        features.append(np.mean(r_peaks))
        features.append(np.std(r_peaks))
    else:
        features += [0, 0, 0, 0]

    return features


(X, Y) = loader.load_all_data()
(Y, mapping) = preprocessing.format_labels(Y)
print('Input length', len(X))
print('Categories mapping', mapping)

NB_EXAMPLES = 4000

subX = X[:NB_EXAMPLES]
subY = Y[:NB_EXAMPLES]

subX = [features_for_row(row) for row in subX]
subY = subY

print(subX[0], subY[0])

from collections import Counter

print("Distribution of categories before balancing")
counter = Counter(subY)
for key in counter.keys():
    print(key, counter[key])

Xt, Xv, Yt, Yv = helper.train_test_split(subX, subY, 0.33)

model = RandomForestClassifier(n_estimators=20, n_jobs=4)
model.fit(Xt, Yt)
print(model.feature_importances_)

Ypredicted = model.predict(Xv)

accuracy = accuracy_score(Yv, Ypredicted)
print(accuracy)
matrix = confusion_matrix(Yv, Ypredicted)
print(matrix)
