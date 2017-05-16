#!/usr/bin/env python3

import argparse
import csv

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

import feature_extractor
import keras_helper as helper
import loader
import preprocessing
import evaluation
from models import *
from utils import logger
from utils.common import set_seed
from utils.system import copy_file


def create_training_set(X, Y):
    return feature_extractor.extract_heartbeats(X, Y)


def get_model(input_shape):
    impl = FCN(input_shape=input_shape)
    return impl


def train(data_dir, model_file):
    (X, Y) = loader.load_all_data(data_dir)
    X = preprocessing.normalize(X)
    Y = preprocessing.format_labels(Y)
    print('Input length', len(X))
    print('Categories mapping', preprocessing.__MAPPING__)
    (X, Y) = create_training_set(X, Y)
    (X, Y) = preprocessing.shuffle_data(X, Y)
    print('Training shape', X.shape)

    impl = get_model(X.shape[1:])
    model = impl.model

    NB_SAMPLES = 50000

    subX = X[:NB_SAMPLES]
    subY = Y[:NB_SAMPLES]

    from collections import Counter

    print("Distribution of categories")
    counter = Counter(subY)
    for key in counter.keys():
        print(key, counter[key])

    Xt, Xv, Yt, Yv = train_test_split(subX, subY, test_size=0.33)

    Yt = to_categorical(Yt, len(preprocessing.__MAPPING__.keys()))
    Yv = to_categorical(Yv, len(preprocessing.__MAPPING__.keys()))

    model_saver = helper.best_model_saver(impl.name())
    learning_optimizer = helper.model_learning_optimizer()
    learning_stopper = helper.learning_stopper()
    model.fit(Xt, Yt,
              epochs=50,
              validation_data=(Xv, Yv),
              callbacks=[
                  model_saver,
                  learning_optimizer,
                  learning_stopper
              ])

    model.load_weights(model_saver.filepath)

    copy_file(model_saver.filepath, model_file)

    evaluation.print_validation_info(Yv, model.predict(Xv), categorical=True)


def classify(file, data_dir, model_file):
    x = loader.load_data_from_file(file, data_dir)
    x = feature_extractor.extract_heartbeats_for_row(x)
    impl = get_model(x.shape[1:])

    model = impl.model
    model.load_weights(model_file)

    return preprocessing.get_original_label(evaluation.from_categorical(model.predict(x)))


def classify_all(data_dir, model_file):
    with open(data_dir + '/RECORDS', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            file_name = row[0]
            label = classify(file_name, data_dir, model_file)

            print(file_name, label)

            with open("answers.txt", "a") as f:
                f.write(file_name + "," + label + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG classifier")
    parser.add_argument("-r", "--record", default="", help="record to classify")
    parser.add_argument("-d", "--dir", default="validation", help="dir with validation files")
    parser.add_argument('--train', dest='train', action='store_true', help="perform training")
    parser.set_defaults(train=False)
    args = parser.parse_args()

    logger.enable_logging('nn', args.train)
    set_seed(42)

    model_file = "weight.h5"

    if args.train:
        train(args.dir, model_file)
    elif len(args.record) > 0:
        label = classify(args.record, args.dir, model_file)

        with open("answers.txt", "a") as f:
            f.write(args.record + "," + label + "\n")
    else:
        if os.path.exists("answers.txt"):
            print("answers.txt already exists, clean it? [y/n]")
            yesno = input().lower().strip()
            if yesno == "yes" or yesno == "y":
                open('answers.txt', 'w').close()
                classify_all(args.dir, model_file)
        else:
            classify_all(args.dir, model_file)
