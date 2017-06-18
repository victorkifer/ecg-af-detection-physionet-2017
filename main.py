import argparse
import csv
from functools import partial
from os import path

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE

from models.dt import RandomForestEcgModel

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
from utils import logger, parallel, matlab
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

    return subX, subY, fn


def get_raw_model(input_shape=None):
    return RandomForestEcgModel()


def get_saved_model(input_shape=None):
    model = get_raw_model(input_shape)
    model.restore()
    return model


def train(data_dir):
    subX, subY, fn = get_training_data(data_dir=data_dir, restore_stored=False)

    Xt, Xv, Yt, Yv = train_test_split(subX, subY, test_size=0.2)

    input_shape = (len(subX[0]),)
    model = get_raw_model(input_shape)
    model.fit(Xt, Yt, validation=(Xv, Yv))
    model.evaluate(Xv, Yv)
    model.show_feature_importances(features_names=fn)


def classify(record, data_dir, clf=None):
    x = loader.load_data_from_file(record, data_dir)
    x = normalizer.normalize_ecg(x)
    x = extractor.features_for_row(x)

    if clf is None:
        clf = get_saved_model(x.shape[1:])

    # as we have one sample at a time to predict, we should resample it into 2d array to classify
    x = np.array(x).reshape(1, -1)

    return categorizer.get_original_label(clf.predict(x)[0])


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

        with open("answers.txt", "a") as f:
            for (name, label) in zip(names, labels):
                f.write(name + "," + label + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG classifier")
    parser.add_argument("-r", "--record", default="", help="record to classify")
    parser.add_argument("-d", "--dir", default="validation", help="dir with validation files")
    parser.add_argument('--train', dest='train', action='store_true', help="perform training")
    parser.set_defaults(train=False)
    args = parser.parse_args()

    logger.enable_logging('ecg', args.train)
    set_seed(42)

    if args.train:
        train(args.dir)
    elif len(args.record) > 0:
        label = classify(args.record, data_dir=args.dir)

        with open("answers.txt", "a") as f:
            f.write(args.record + "," + label + "\n")
    else:
        if path.exists("answers.txt"):
            print("answers.txt already exists, clean it? [y/n]")
            yesno = input().lower().strip()
            if yesno == "yes" or yesno == "y":
                open('answers.txt', 'w').close()
                classify_all(args.dir)
        else:
            classify_all(args.dir)
