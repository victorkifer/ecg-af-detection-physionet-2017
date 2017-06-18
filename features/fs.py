from scipy.fftpack import fft


def extract_fft(x):
    return fft(x)[:len(x) // 2]