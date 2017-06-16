import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

from biosppy.signals import ecg
from features import hrv
from features.melbourne_eeg import calcActivity, calcMobility, calcComplexity
from loading import loader


def frequency_powers(x, n_power_features=40):
    fxx, pxx = signal.welch(x, loader.FREQUENCY)
    features = dict()
    for i, v in enumerate(pxx[:n_power_features]):
        features['welch' + str(i)] = v

    return features


def frequency_powers_summary(x):
    ecg_fs_range = (0, 50)
    band_size = 5

    features = dict()

    fxx, pxx = signal.welch(x, loader.FREQUENCY)
    for i in range((ecg_fs_range[1] - ecg_fs_range[0]) // 5):
        fs_min = i * band_size
        fs_max = fs_min + band_size
        indices = np.logical_and(fxx >= fs_min, fxx < fs_max)
        bp = np.sum(pxx[indices])
        features["power_" + str(fs_min) + "_" + str(fs_max)] = bp

    return features


def heart_rate_features(hr):
    features = {
        'hr_max': 0,
        'hr_min': 0,
        'hr_mean': 0,
        'hr_std': 0
    }

    if len(hr) > 0:
        features['hr_max'] = np.amax(hr)
        features['hr_min'] = np.amin(hr)
        features['hr_mean'] = np.mean(hr)
        features['hr_std'] = np.std(hr)

    return features


def heart_beats_features(thb):
    means = np.array([col.median() for col in thb.T])
    mins = np.array([col.min() for col in thb.T])
    maxs = np.array([col.max() for col in thb.T])
    # stds = np.array([col.std() for col in thb.T])

    features = dict()
    for i, v in enumerate(means):
        features['median' + str(i)] = v

    for i, v in enumerate(mins):
        features['min' + str(i)] = v

    for i, v in enumerate(maxs):
        features['max' + str(i)] = v

    return features


def heart_beats_features2(thb):
    means = np.array([np.median(col) for col in thb.T])
    stds = np.array([np.std(col) for col in thb.T])

    r_pos = int(0.2 * loader.FREQUENCY)

    if means[r_pos] < 0:
        means *= -1

    PQ = means[:int(0.15 * loader.FREQUENCY)]
    ST = means[int(0.25 * loader.FREQUENCY):]

    QR = means[int(0.13 * loader.FREQUENCY):r_pos]
    RS = means[r_pos:int(0.27 * loader.FREQUENCY)]

    q_pos = int(0.13 * loader.FREQUENCY) + np.argmin(QR)
    s_pos = r_pos + np.argmin(RS)

    p_pos = np.argmax(PQ)
    t_pos = np.argmax(ST)

    t_wave = ST[max(0, t_pos - int(0.08 * loader.FREQUENCY)):min(len(ST), t_pos + int(0.08 * loader.FREQUENCY))]
    p_wave = PQ[max(0, p_pos - int(0.06 * loader.FREQUENCY)):min(len(PQ), p_pos + int(0.06 * loader.FREQUENCY))]

    QRS = means[q_pos:s_pos]

    a = dict()
    a['PR_interval'] = r_pos - p_pos
    a['P_max'] = PQ[p_pos]
    a['P_to_R'] = PQ[p_pos] / means[r_pos]
    a['ST_interval'] = t_pos
    a['T_max'] = ST[t_pos]
    a['T_to_R'] = ST[t_pos] / means[r_pos]
    a['P_to_T'] = PQ[p_pos] / ST[t_pos]
    a['P_skew'] = skew(p_wave)
    a['P_kurt'] = kurtosis(p_wave)
    a['T_skew'] = skew(t_wave)
    a['T_kurt'] = kurtosis(t_wave)
    a['activity'] = calcActivity(means)
    a['mobility'] = calcMobility(means)
    a['complexity'] = calcComplexity(means)
    a['QRS_len'] = s_pos - q_pos
    a['QS_diff'] = abs(means[s_pos] - means[q_pos])
    a['QRS_kurt'] = kurtosis(QRS)
    a['QRS_skew'] = skew(QRS)
    a['P_std'] = np.mean(stds[:q_pos])
    a['T_std'] = np.mean(stds[s_pos:])

    for i in range(len(means)):
        a['median_' + str(i)] = means[i]

    return a


def heart_beats_features3(thb):
    means = np.array([np.mean(col) for col in thb.T])
    medians = np.array([np.median(col) for col in thb.T])

    diff = np.subtract(means, medians)
    diff = np.power(diff, 2)

    return {
        'mean_median_diff_mean': np.mean(diff),
        'mean_median_diff_std': np.std(diff)
    }


def r_features(s, r_peaks):
    r_vals = [s[i] for i in r_peaks]

    times = np.diff(r_peaks)
    avg = np.mean(times)
    filtered = sum([1 if i < 0.5 * avg else 0 for i in times])

    total = len(r_vals) if len(r_vals) > 0 else 1

    data = hrv.time_domain(times)

    data['beats_to_length'] = len(r_peaks) / len(s)
    data['r_mean'] = np.mean(r_vals)
    data['r_std'] = np.std(r_vals)
    data['filtered_r'] = filtered
    data['rel_filtered_r'] = filtered / total

    return data


def add_suffix(dic, suffix):
    keys = list(dic.keys())
    for key in keys:
        dic[key + suffix] = dic.pop(key)
    return dic


def get_features_dict(x):
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

    fx = dict()
    fx.update(heart_rate_features(hr))
    # fx.update(frequency_powers(x))
    fx.update(frequency_powers(fts))
    fx.update(frequency_powers_summary(fts))
    fx.update(heart_beats_features2(thb))
    # fx.update(heart_beats_features3(thb))
    fx.update(r_features(fts, rpeaks))

    fx['PRbyST'] = fx['PR_interval'] * fx['ST_interval']
    fx['P_form'] = fx['P_kurt'] * fx['P_skew']
    fx['T_form'] = fx['T_kurt'] * fx['T_skew']

    return fx


def get_feature_names(x):
    features = get_features_dict(x)
    return sorted(list(features.keys()))


def features_for_row(x):
    features = get_features_dict(x)
    return [features[key] for key in sorted(list(features.keys()))]
