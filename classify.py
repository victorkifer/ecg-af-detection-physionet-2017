import argparse

import feature_extractor
import loader
from models import *


def classify(record_dir, record, weights):
    x = loader.load_data_from_file(record, record_dir)

    x = feature_extractor.extract_heartbeats_for_row(x)

    # This is required for FCN
    x = x.reshape((1, 1, x.shape[1]))
    print(x.shape)

    impl = FCN(input_shape=x.shape[1:])
    model = impl.model
    model.summary()

    model.load_weights(weights)

    Y = model.predict(x)

    mapping = {
        0: "A",
        1: "N",
        2: "O",
        3: "~"
    }

    with open("answers.txt", "a") as f:
        f.write(record + "," + mapping[Y] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG classifier")
    parser.add_argument("-r", "--record", help="record to classify")
    parser.add_argument("-d", "--dir", default="validation", help="dir with validation files")
    parser.add_argument("-w", "--weights", default="weights.h5", help="weights to load")
    args = parser.parse_args()

    RECORD_FILE = args.record
    WEIGHTS_FILE = args.weights
    VALIDATION_DIR = args.dir
