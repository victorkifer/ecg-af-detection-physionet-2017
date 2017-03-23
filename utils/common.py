import random
import numpy as np


def set_seed(seed=None):
    if seed is None:
        seed = int(random.random() * 1e6)
    random.seed(seed)
    np.random.seed(seed)
    print("Seed =", seed)
