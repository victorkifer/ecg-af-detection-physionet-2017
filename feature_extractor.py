import multiprocessing as mp

from pywt import wavedec

from qrs_detect import *
from qrs_detect2 import qrs_detect2

FREQUENCY = 300
MIN_HEARTBEAT_TIME = int(0.4 * FREQUENCY)

PR_TIME = int(0.2 * FREQUENCY)
QRS_TIME = int(0.1 * FREQUENCY)
ST_TIME = int(0.3 * FREQUENCY)

BEFORE_R = PR_TIME + QRS_TIME // 2
AFTER_R = ST_TIME + QRS_TIME // 2


def trimboth(row, portion):
    filter = portion * max(
        fabs(np.amin(row)),
        abs(np.amax(row))
    )

    return np.array([x if -filter < x < filter else 0 for x in row])


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
    r = get_r_peaks_positions(ecg, fs=300, thres=0.4, ref_period=0.2)
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


def extract_pqrst(row):
    PR = 0.16
    QRS = 0.1
    QT = 0.44
    QR = QRS / 2
    RS = QRS / 2
    ST = QT - QRS
    PQ = PR - QR

    PR = int(PR * FREQUENCY)
    QRS = int(QRS * FREQUENCY)
    QT = int(QT * FREQUENCY)
    QR = int(QR * FREQUENCY)
    RS = int(RS * FREQUENCY)
    ST = int(ST * FREQUENCY)
    PQ = int(PQ * FREQUENCY)

    r = get_r_peaks_positions(row)
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


def get_r_peaks_positions(row):
    return qrs_detect(row)
    # return qrs_detect2(row, fs=300, thres=0.4, ref_period=0.2)
