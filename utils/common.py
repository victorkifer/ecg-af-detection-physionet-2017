import random
import numpy as np

from scipy import stats

def set_seed(seed=None):
    if seed is None:
        seed = int(random.random() * 1e6)
    random.seed(seed)
    np.random.seed(seed)
    print("Seed =", seed)


def mode(a):
    return stats.mode(a, axis=None)[0][0]