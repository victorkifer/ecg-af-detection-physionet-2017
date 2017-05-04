import numpy as np
from biosppy.signals import ecg
from scipy import signal
from scipy.stats import skew, kurtosis

import loader
from fft import compute_fft
from melbourne_eeg import calcActivity, calcMobility, calcComplexity
from utils import common, matlab


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def filter_peaks(ts, fts, rpeaks, tts, thb, hrts, hr):
    rri = np.diff(rpeaks)
    rr_mean = np.mean(rri)

    misclassified = [x+1 for x in matlab.find(rri, lambda x: x < 0.5 * rr_mean)]
    normalized = group_consecutives(misclassified)
    misclassified = []
    for item in normalized:
        if type(item) is list:
            misclassified += item[::2]
        else:
            misclassified.append(item)
    print(misclassified)

    rpeaks = np.delete(rpeaks, misclassified)
    thb = np.delete(thb, misclassified)
    hr = np.delete(hr, misclassified)

    return ts, fts, rpeaks, tts, thb, hrts, hr




def frequency_powers(x, n_power_features=30):
    f, prd = signal.welch(x, loader.FREQUENCY)
    return prd[:n_power_features]


def heart_rate_features(hr):
    zhb = np.zeros(4)
    if len(hr) > 0:
        zhb[0] = np.amax(hr)
        zhb[1] = np.amin(hr)
        zhb[2] = np.mean(hr)
        zhb[3] = np.std(hr)
    return zhb


def heart_beats_features(thb):
    means = np.array([col.mean() for col in thb.T])
    stds = np.array([col.std() for col in thb.T])

    return np.concatenate([means, stds])


def heart_beats_features2(thb):
    means = np.array([col.mean() for col in thb.T])
    stds = np.array([col.std() for col in thb.T])

    PQ = means[:int(0.15 * loader.FREQUENCY)]
    ST = means[int(0.25 * loader.FREQUENCY):]
    r_pos = int(0.2 * loader.FREQUENCY)

    a = np.zeros(15)
    p_pos = np.argmax(PQ)
    t_pos = np.argmax(ST)
    a[0] = r_pos - p_pos
    a[1] = PQ[p_pos]
    a[2] = PQ[p_pos] / means[r_pos]
    a[3] = t_pos
    a[4] = ST[t_pos]
    a[5] = ST[t_pos] / means[r_pos]
    a[6] = PQ[p_pos] / ST[t_pos]
    a[7] = skew(PQ)
    a[8] = kurtosis(PQ)
    a[9] = skew(ST)
    a[10] = kurtosis(ST)
    a[11] = calcActivity(means)
    a[12] = calcMobility(means)
    a[13] = calcComplexity(means)
    a[14] = np.mean(stds)

    return a


def heart_beats_features3(thb):
    means = np.array([col.mean() for col in thb.T])
    medians = np.array([common.mode(col) for col in thb.T])

    diff = np.subtract(means, medians)
    diff = np.power(diff, 2)

    return np.array([diff.mean()])


def features_for_row(x):
    [ts, fts, rpeaks, tts, thb, hrts, hr] = ecg.ecg(signal=x, sampling_rate=loader.FREQUENCY, show=False)
    """
    Returns:	

    ts (array) – Signal time axis reference (seconds).
    filtered (array) – Filtered ECG signal.
    rpeaks (array) – R-peak location indices.
    templates_ts (array) – Templates time axis reference (seconds).
    templates (array) – Extracted heartbeat templates.
    heart_rate_ts (array) – Heart rate time axis reference (seconds).
    heart_rate (array) – Instantaneous heart rate (bpm).
    """

    return np.concatenate([
        heart_rate_features(hr),
        frequency_powers(x),
        heart_beats_features2(thb),
        heart_beats_features3(thb)
    ])
