import numpy as np
from scipy import interpolate
from scipy.signal import welch


def time_domain(rri):
    """
    Computes time domain characteristics of heart rate
    :param rri: RR intervals in ms
    :return:
    """
    rmssd = 0
    sdnn = 0
    nn50 = 0
    pnn50 = 0
    mrri = 0
    stdrri = 0
    mhr = 0

    if len(rri) > 0:
        diff_rri = np.diff(rri)
        if len(diff_rri) > 0:
            # Root mean square of successive differences
            rmssd = np.sqrt(np.mean(diff_rri ** 2))
            # Number of pairs of successive NNs that differe by more than 50ms
            nn50 = sum(abs(diff_rri) > 50)
            # Proportion of NN50 divided by total number of NNs
            pnn50 = (nn50 / len(diff_rri)) * 100

        # Standard deviation of NN intervals
        sdnn = np.std(rri, ddof=1)  # make it calculates N-1
        # Mean of RR intervals
        mrri = np.mean(rri)
        # Std of RR intervals
        stdrri = np.std(rri)
        # Mean heart rate, in ms
        mhr = 60 * 1000.0 / mrri

    keys = ['rmssd', 'sdnn', 'nn50', 'pnn50', 'mrri', 'stdrri', 'mhr']
    values = [rmssd, sdnn, nn50, pnn50, mrri, stdrri, mhr]
    values = np.round(values, 2)
    values = np.nan_to_num(values)

    return dict(zip(keys, values))


def _interpolate_rri(rri, interp_freq=4):
    time_rri = _create_time_info(rri)
    time_rri_interp = np.arange(0, time_rri[-1], 1 / interp_freq)
    tck = interpolate.splrep(time_rri, rri, s=0)
    rri_interp = interpolate.splev(time_rri_interp, tck, der=0)
    return time_rri_interp, rri_interp


def _create_time_info(rri):
    rri_time = np.cumsum(rri) / 1000.0  # make it seconds
    return rri_time - rri_time[0]  # force it to start at zero


def _transform_rri_to_milliseconds(rri):
    if np.median(rri) < 1:
        rri *= 1000
    return rri


def frequency_domain(rri, interp_freq=4):
    """
    Interpolates a signal and performs Welch estimation of power spectrum
    :param rri:
    :param interp_freq:
    :return:
    """

    fxx = np.array([])
    pxx = np.array([])

    if len(rri) >= 4:
        # frequency domain cannot be computed on short RRi
        time_interp, rri_interp = _interpolate_rri(rri, interp_freq)
        fxx, pxx = welch(x=rri_interp, fs=interp_freq, nperseg=rri_interp.shape[-1])

    return _bands_energy(fxx, pxx)


def _bands_energy(fxx, pxx, vlf_band=(0, 0.04), lf_band=(0.04, 0.15),
                  hf_band=(0.15, 0.4)):
    vlf_indexes = np.logical_and(fxx >= vlf_band[0], fxx < vlf_band[1])
    lf_indexes = np.logical_and(fxx >= lf_band[0], fxx < lf_band[1])
    hf_indexes = np.logical_and(fxx >= hf_band[0], fxx < hf_band[1])

    if len(vlf_indexes) > 0:
        vlf = np.trapz(y=pxx[vlf_indexes], x=fxx[vlf_indexes])
    else:
        vlf = 0

    if len(lf_indexes) > 0:
        lf = np.trapz(y=pxx[lf_indexes], x=fxx[lf_indexes])
    else:
        lf = 0

    if len(hf_indexes) > 0:
        hf = np.trapz(y=pxx[hf_indexes], x=fxx[hf_indexes])
    else:
        hf = 0

    total_power = vlf + lf + hf

    if hf != 0:
        lf_hf = lf / hf
    else:
        lf_hf = 0

    if total_power - vlf != 0:
        lfnu = (lf / (total_power - vlf)) * 100
        hfnu = (hf / (total_power - vlf)) * 100
    else:
        lfnu = 0
        hfnu = 0

    keys = ['total_power', 'vlf', 'lf', 'hf', 'lf_hf', 'lfnu', 'hfnu']
    values = [total_power, vlf, lf, hf, lf_hf, lfnu, hfnu]
    values = np.round(values, 2)
    values = np.nan_to_num(values)

    return dict(zip(keys, values))
