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


def extract_heart_beats(row):
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
    return peakutils.indexes(row, thres=0.6, min_dist=MIN_HEARTBEAT_TIME)


def get_r_peaks_frequencies(row):
    peaks = get_r_peaks_positions(row)
    times = []
    prev = peaks[0]
    for peak in peaks[1:]:
        times.append(peak - prev)
        prev = peak
    return times
