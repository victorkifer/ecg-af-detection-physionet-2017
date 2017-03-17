import argparse

import feature_extractor
import loader
from models import *

parser = argparse.ArgumentParser(description="ECG classifier")
parser.add_argument("-r", "--record", help="record to classify")
parser.add_argument("-w", "--weights", default="weights.h5", help="weights to load")
args = parser.parse_args()

RECORD_FILE = args.record
WEIGHTS_FILE = args.weights

X = loader.load_data_from_file(RECORD_FILE, "validation")

X = feature_extractor.extract_heartbeats_for_row(X)

# This is required for FCN
X = X.reshape((1, 1, X.shape[1]))
print(X.shape)

impl = FCN(input_shape=X.shape[1:])
model = impl.model
model.summary()

model.load_weights(WEIGHTS_FILE)

Y = model.predict(X)

mapping = {
    0: "A",
    1: "N",
    2: "O",
    3: "~"
}

with open("answers.txt", "a") as f:
    f.write(RECORD_FILE + "," + mapping[Y] + "\n")
