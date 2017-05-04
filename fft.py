import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from sklearn.preprocessing import normalize

import loader


def compute_fft(row):
    fft_embedding_size = 400

    wf = normalize(row.reshape(1, -1))

    wf_fft = np.abs(fft(wf))
    wf_fft = wf_fft[:, :fft_embedding_size].reshape(-1)

    return wf_fft
