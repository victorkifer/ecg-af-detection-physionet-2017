#!/usr/bin/env python3

import argparse
import csv

from scipy.stats import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score

import evaluation
import loader
import preprocessing
from common.qrs_detect import *
from feature_extractor import wavelet_coefficients, extract_pqrst
from utils import async
from utils import common
from utils import logger
from utils.common import set_seed

NB_RR = 60


def features_for_row(row):
    features = []

    features += wavelet_coefficients(row)

    pqrsts = extract_pqrst(row)

    features.append(len(pqrsts) * 1.0 * loader.FREQUENCY / len(row))

    if len(pqrsts) == 0:
        return features + [0 for x in range(5 + NB_RR * 12)]

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

    pqrsts = pqrsts[:min(NB_RR, len(pqrsts))]
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
            common.mode(pq),

            # feature for ST interval
            tmax,
            tmax / row[r[i]],
            np.mean(st),
            np.std(st),
            common.mode(st),

            # features for whole PQRST interval
            stats.skew(pt),
            stats.kurtosis(pt)
        ]

    for i in range(NB_RR - len(pqrsts)):
        features += [0 for x in range(12)]

    return features


def train(data_dir, model_file):
    (X, Y) = loader.load_all_data(data_dir)
    X = preprocessing.normalize(X)
    Y = preprocessing.format_labels(Y)
    print('Input length', len(X))
    print('Categories mapping', preprocessing.__MAPPING__)

    subX = X
    subY = Y

    print("Features extraction started")
    subX = async.apply_async(subX, features_for_row)

    np.savez('outputs/processed.npz', x=subX, y=subY)

    print("Features extraction finished")
    subY = subY

    from collections import Counter

    print("Distribution of categories before balancing")
    counter = Counter(subY)
    for key in counter.keys():
        print(key, counter[key])

    Xt, Xv, Yt, Yv = train_test_split(subX, subY, test_size=0.33)

    model = RandomForestClassifier(n_estimators=20, n_jobs=async.get_number_of_jobs())
    model.fit(Xt, Yt)
    print(model.feature_importances_)

    joblib.dump(model, model_file)

    Ypredicted = model.predict(Xv)

    evaluation.print_validation_info(Yv, Ypredicted)


def load_model(model_file):
    return joblib.load(model_file)


def classify(model, record, data_dir):
    x = loader.load_data_from_file(record, data_dir)
    x = normalize_ecg(x)
    x = features_for_row(x)

    # as we have one sample at a time to predict, we should resample it into 2d array to classify
    x = np.array(x).reshape(1, -1)

    return preprocessing.get_original_label(model.predict(x)[0])


def classify_all(data_dir, model_file):
    model = joblib.load(model_file)
    with open(data_dir + '/RECORDS', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            file_name = row[0]
            label = classify(model, file_name, data_dir)

            print(file_name, label)

            with open("answers.txt", "a") as f:
                f.write(file_name + "," + label + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG classifier")
    parser.add_argument("-r", "--record", default=None, help="record to classify")
    parser.add_argument("-d", "--dir", default="validation", help="dir with validation files")
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    if args.train:
        logger.enable_logging('ml', True)
    else:
        logger.enable_logging('ml', False)

    set_seed(42)

    model_file = "model.pkl"

    if args.train:
        train(args.dir, model_file)
    elif args.record is not None:
        model = joblib.load(model_file)
        label = classify(model, args.record, args.dir)

        with open("answers.txt", "a") as f:
            f.write(args.record + "," + label + "\n")
    else:
        classify_all(args.dir, model_file)
