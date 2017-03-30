from pywt import wavedec
from scipy.stats import stats

import loader

import numpy as np

from common.qrs_detect import *
from preprocessing import trimboth
from utils import common
from utils import matlab

NB_RR = 80


def extract_pqrst(row):
    PR = 0.25
    QRS = 0.1
    QT = 0.44
    QR = QRS / 2
    RS = QRS / 2
    ST = QT - QRS
    PQ = PR - QR

    PR = int(PR * loader.FREQUENCY)
    QRS = int(QRS * loader.FREQUENCY)
    QT = int(QT * loader.FREQUENCY)
    QR = int(QR * loader.FREQUENCY)
    RS = int(RS * loader.FREQUENCY)
    ST = int(ST * loader.FREQUENCY)
    PQ = int(PQ * loader.FREQUENCY)

    r = qrs_detect(row)
    beats = []
    for R in r:
        start = R - PR
        if start < 0:
            continue
        end = R + RS + ST
        if end > len(row):
            continue

        pqrst = (R - PR, R - QR, R, R + RS, R + RS + ST)
        beats.append(pqrst)
    return beats


def wavelet_coefficients(row):
    a, d1, d2 = wavedec(row, 'db1', level=2)

    d1n = trimboth(d1, 0.1)
    d2n = trimboth(d2, 0.1)

    m1 = np.mean(d1)
    s1 = np.std(d1)
    m1n = np.mean(d1n)
    s1n = np.std(d1n)

    m2 = np.mean(d2)
    s2 = np.std(d2)
    m2n = np.mean(d2n)
    s2n = np.std(d2n)

    return [
        m1n,
        s1n,
        abs(m1n) - abs(m1),
        abs(s1n) - abs(s1),
        m2n,
        s2n,
        abs(m2n) - abs(m2),
        abs(s2n) - abs(s2)
    ]


def rr_diff(rrs):
    rr_d = []
    rr_sqd = []

    for i in range(1, len(rrs)):
        rr_d.append(abs(rrs[i - 1] - rrs[i]))
        rr_sqd.append(pow(rrs[i - 1] - rrs[i], 2))

    return rr_d, rr_sqd


def features_for_row(row):
    features = []

    features += wavelet_coefficients(row)

    pqrsts = extract_pqrst(row)

    features.append(len(pqrsts) * 1.0 * loader.FREQUENCY / len(row))

    if len(pqrsts) == 0:
        return features + [0 for x in range(10 + NB_RR * 13)]

    p = [x[0] for x in pqrsts]
    q = [x[1] for x in pqrsts]
    r = [x[2] for x in pqrsts]
    s = [x[3] for x in pqrsts]
    t = [x[4] for x in pqrsts]

    r_val = [row[i] for i in r]
    features.append(np.mean(r_val))
    features.append(np.std(r_val))

    rrs = np.diff(r)

    if len(rrs) > 0:
        mean = np.mean(rrs)
        original_len = len(rrs)
        rrs = matlab.select(rrs, lambda x: 0.5 * mean < x < 1.5 * mean)
        features.append(len(rrs) / original_len)
    else:
        features.append(0)

    if len(rrs) > 0:
        (rr_d, rr_sqd) = rr_diff(rrs)
        sdsd = 0
        rmssd = 0
        if len(rr_d) > 0:
            sdsd = np.std(rr_d)
            rmssd = np.sqrt(np.mean(rr_sqd))

        features += [
            np.amin(rrs),
            np.amax(rrs),
            np.mean(rrs),
            np.std(rrs),
            sdsd,
            rmssd,
            sum([1 for x in r if x < 0])
        ]
    else:
        features += [0 for x in range(7)]

    pqrsts = pqrsts[:min(NB_RR, len(pqrsts))]
    row = low_pass_filtering(row)
    row = high_pass_filtering(row)
    for i in range(len(pqrsts)):
        pq = row[p[i]:q[i]]
        st = row[s[i]:t[i]]
        pt = row[p[i]:t[i]]
        pmax = np.amax(pq)
        tmax = np.amax(st)

        p_mean = np.mean(pq)
        t_mean = np.mean(st)

        features += [
            # features for PQ interval
            pmax,
            pmax / row[r[i]],
            p_mean,
            np.std(pq),
            common.mode(pq),

            # feature for ST interval
            tmax,
            tmax / row[r[i]],
            t_mean,
            np.std(st),
            common.mode(st),

            p_mean / t_mean,

            # features for whole PQRST interval
            stats.skew(pt),
            stats.kurtosis(pt)
        ]

    for i in range(NB_RR - len(pqrsts)):
        features += [0 for x in range(13)]

    return features
