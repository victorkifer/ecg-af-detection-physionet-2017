import multiprocessing as mp

import numpy as np
from pywt import wavedec

from common.qrs_detect import *

FREQUENCY = 300
MIN_HEARTBEAT_TIME = int(0.4 * FREQUENCY)

PR_TIME = int(0.2 * FREQUENCY)
QRS_TIME = int(0.1 * FREQUENCY)
ST_TIME = int(0.3 * FREQUENCY)

BEFORE_R = PR_TIME + QRS_TIME // 2
AFTER_R = ST_TIME + QRS_TIME // 2


def extract_heartbeats(X, Y):
    pool = mp.Pool()
    x_new = pool.map(extract_heartbeats_for_row, X)
    pool.close()
    x_out = []
    y_out = []
    for i in range(len(x_new)):
        out = x_new[i]
        y = Y[i]
        for o in out:
            x_out.append(o)
            y_out.append(y)
    return np.array(x_out), y_out


def extract_heartbeats_for_row(ecg):
    r = get_r_peaks_positions(ecg)
    beats = []
    for peak in r:
        start = peak - BEFORE_R
        end = peak + AFTER_R
        if start < 0:
            continue
        if end > len(ecg):
            continue
        beats.append(ecg[start:end])
    return beats


def get_r_peaks_positions(row):
    return qrs_detect(row)
    # return qrs_detect2(row, fs=300, thres=0.4, ref_period=0.2)
