#!/usr/bin/env python3

import argparse
import csv
from functools import partial
from os import path

from models.dt import RandomForestEcgModel
from models.nn import *

try:
    import matplotlib

    matplotlib.use("Qt5Agg")
except ImportError:
    print("Matplotlib is not installed")

import numpy as np
from sklearn.model_selection import train_test_split

from features import feature_extractor5
from loading import loader
from preprocessing import categorizer, balancer, normalizer
from utils import logger, parallel
from utils.common import set_seed, shuffle_data

extractor = feature_extractor5


def get_training_data(data_dir=None, restore_stored=False):
    if restore_stored:
        file = np.load('outputs/processed.npz')
        subX = file['x']
        subY = file['y']
        fn = file['fn']
    else:
        x, y = loader.load_all_data(data_dir)
        x, y = shuffle_data(x, y)
        y = categorizer.format_labels(y)
        print('Categories mapping', categorizer.__MAPPING__)
        print('Input length', len(x))
        print("Distribution of categories before balancing")
        balancer.show_balancing(y)

        subX, subY = balancer.balance2(x, y)

        subX = normalizer.normalize_batch(subX)
        print('Input length', len(subX))
        print("Distribution of categories after balancing")
        balancer.show_balancing(subY)

        print("Features extraction started")
        fn = extractor.get_feature_names(subX[0])
        subX = parallel.apply_async(subX, extractor.features_for_row)

        np.savez('outputs/processed.npz', x=subX, y=subY, fn=fn)

    print("Features extraction finished", len(subX[0]))

    print("Feature names")
    for i, n in enumerate(fn):
        print((i, n))

    return np.array(subX), np.array(subY), fn


def get_raw_model(input_shape=None):
    return FcnEcgModel(input_shape)


def get_saved_model(input_shape=None):
    model = get_raw_model(input_shape)
    model.restore()
    return model


def train(args):
    subX, subY, fn = get_training_data(data_dir=args.dir, restore_stored=True)

    Xt, Xv, Yt, Yv = train_test_split(subX, subY, test_size=0.2)

    input_shape = subX.shape[1:]
    model = get_raw_model(input_shape)
    model.fit(Xt, Yt, validation=(Xv, Yv))
    model.evaluate(Xv, Yv)
    # model.show_feature_importances(features_names=fn)


def classify(record, data_dir, clf=None):
    x = loader.load_data_from_file(record, data_dir)
    x = normalizer.normalize_ecg(x)
    x = extractor.features_for_row(x)

    if clf is None:
        clf = get_saved_model(x.shape[1:])

    # as we have one sample at a time to predict, we should resample it into 2d array to classify
    x = np.array(x).reshape(1, -1)

    return categorizer.get_original_label(clf.predict(x)[0])


def format_result(record, label):
    return record + "," + label


def classify_all(data_dir):
    model = get_saved_model()
    print("Model is loaded")
    with open(data_dir + '/RECORDS', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        print("Starting classification")
        func = partial(classify, data_dir=data_dir, clf=model)
        names = [row[0] for row in reader]
        labels = parallel.apply_async(names, func)
        print("Classification finished, saving results")

        with open(args.output, "a") as f:
            for (name, label) in zip(names, labels):
                print(format_result(name, label))
                f.write(format_result(name, label) + "\n")


def main_classify_single(args):
    label = classify(args.record, data_dir=args.dir)

    with open(args.output, "a") as f:
        print(format_result(args.record, label))
        f.write(format_result(args.record, label) + "\n")


def main_classify_all(args):
    if path.exists(args.output):
        print(args.output + " already exists, clean it? [y/n]")
        yesno = input().lower().strip()
        if yesno == "yes" or yesno == "y":
            open(args.output, 'w').close()
            classify_all(args.dir)
    else:
        classify_all(args.dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG classifier")
    parser.add_argument("-d", "--dir", default="data", help="Directory with the data")
    parser.add_argument("-m", "--mode", help="Script working mode. One of [train, classify]")
    parser.add_argument("-r", "--record", default="", help="Name of the record to be classified")
    parser.add_argument("-o", "--output", default="answers.txt", help="File where to write classification results")
    args = parser.parse_args()

    set_seed(42)

    if args.mode == "train":
        logger.enable_logging('ecg', True)
        train(args)
    elif args.mode == "classify":
        if len(args.record) > 0:
            main_classify_single(args)
        else:
            main_classify_all(args)
    else:
        parser.print_help()
