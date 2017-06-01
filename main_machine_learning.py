#!/usr/bin/env python3

import feature_extractor5

try:
    import matplotlib

    matplotlib.use("Qt5Agg")
except ImportError:
    print("Matplotlib is not installed")

import argparse
import csv
import os
from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score

import evaluation
import loader
import preprocessing
import tree_info
from common.qrs_detect import *
from utils import async
from utils import logger
from utils.common import set_seed


def train(data_dir, model_file):
    (X, Y) = loader.load_all_data(data_dir)
    Y = preprocessing.format_labels(Y)
    print('Categories mapping', preprocessing.__MAPPING__)
    print('Input length', len(X))
    print("Distribution of categories before balancing")
    preprocessing.show_balancing(Y)

    X, Y = preprocessing.balance2(X, Y)

    subX = X
    subY = Y

    subX = preprocessing.normalize(subX)
    print('Input length', len(subX))
    print("Distribution of categories after balancing")
    preprocessing.show_balancing(subY)

    print("Features extraction started")
    subX = async.apply_async(subX, feature_extractor5.features_for_row)

    np.savez('outputs/processed.npz', x=subX, y=subY)

    # file = np.load('outputs/processed.npz')
    # subX = file['x']
    # subY = file['y']

    print("Features extraction finished", len(subX[0]))
    subY = subY

    Xt, Xv, Yt, Yv = train_test_split(subX, subY, test_size=0.2)

    model = RandomForestClassifier(n_estimators=60, n_jobs=async.get_number_of_jobs())
    scores = cross_val_score(model, subX, subY, cv=5)
    print('Cross-Validation', scores, scores.mean())

    model = RandomForestClassifier(n_estimators=60, n_jobs=async.get_number_of_jobs())
    model.fit(Xt, Yt)
    joblib.dump(model, model_file)
    Ypredicted = model.predict(Xv)

    evaluation.print_validation_info(Yv, Ypredicted)
    tree_info.plot_tree_info(model)


def load_model(model_file):
    return joblib.load(model_file)


def classify(record, clf, data_dir):
    x = loader.load_data_from_file(record, data_dir)
    x = preprocessing.normalize_ecg(x)
    x = feature_extractor5.features_for_row(x)

    # as we have one sample at a time to predict, we should resample it into 2d array to classify
    x = np.array(x).reshape(1, -1)

    return preprocessing.get_original_label(clf.predict(x)[0])


def classify_all(data_dir, model_file):
    model = joblib.load(model_file)
    print("Model is loaded")
    with open(data_dir + '/RECORDS', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        print("Starting classification")
        func = partial(classify, clf=model, data_dir=data_dir)
        names = [row[0] for row in reader]
        labels = async.apply_async(names, func)
        print("Classification finished, saving results")

        with open("answers.txt", "a") as f:
            for (name, label) in zip(names, labels):
                f.write(name + "," + label + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG classifier")
    parser.add_argument("-r", "--record", default=None, help="record to classify")
    parser.add_argument("-d", "--dir", default="validation", help="dir with validation files")
    parser.add_argument('--train', dest='train', action='store_true')
    parser.set_defaults(train=False)
    args = parser.parse_args()

    logger.enable_logging('ml', args.train)
    set_seed(42)

    model_file = "model.pkl"

    if args.train:
        train(args.dir, model_file)
    elif args.record is not None:
        model = joblib.load(model_file)
        label = classify(args.record, model, args.dir)

        with open("answers.txt", "a") as f:
            f.write(args.record + "," + label + "\n")
        print(args.record + "," + label)
    else:
        if os.path.exists("answers.txt"):
            print("answers.txt already exists, clean it? [y/n]")
            yesno = input().lower().strip()
            if yesno == "yes" or yesno == "y":
                open('answers.txt', 'w').close()
                classify_all(args.dir, model_file)
        else:
            classify_all(args.dir, model_file)
