import multiprocessing as mp

import numpy as np
import peakutils

from qrs_detect import qrs_detect, cancel_dc_drift, qrs_detect_normalized

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


def extract_heartbeats_for_row(row):
    row = cancel_dc_drift(row)
    q,r,s = qrs_detect_normalized(row)
    beats = []
    for peak in r:
        start = peak - BEFORE_R
        end = peak + AFTER_R
        if start < 0:
            continue
        if end > len(row):
            continue
        beats.append(row[start:end])
    return beats


def get_r_peaks_positions(row):
    row = cancel_dc_drift(row)
    q, r, s = qrs_detect_normalized(row)
    return r


def get_r_peaks_frequencies(row):
    peaks = get_r_peaks_positions(row)
    times = []
    prev = peaks[0]
    for peak in peaks[1:]:
        times.append(peak - prev)
        prev = peak
    return times
