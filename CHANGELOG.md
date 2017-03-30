# CHANGELOG

## Entry 2

Pre-processing:

- Removed DC component (ecg - mean(ecg))
- Normalized in range of [-1; 1]

The second entry features vector consists of:

- wavelet coefficients
    - mean(wavelet details 1)
    - std(wavelet details 1)
    - mean(wavelet details 2)
    - std(wavelet details 2)
- number of r peaks divided by len of the record
- mean of r peak values
- std of r peak values
- proportion of length of normalized rri (0.5 * mean < rri < 1.5 * mean) divided by the length of origin rri
- hrv computed on normalized rri:
    - min rri
    - max rri
    - mean rri
    - std rri
    - SDSD ("standard deviation of successive differences")
    - RMSSD ("root mean square of successive differences")
- number of negative r values

The next features are computed on the ecg with additional preprocessing:
- low pass filter
- high pass filter
- derivative filter
- squaring
- integrated moving window

This list of features was computed for first 80 PQRST:
- max value of PQ interval
- max value of PQ interval divided by value of R
- mean value of PQ interval
- std of PQ
- mode of PQ
- max value of ST interval
- max value of ST interval divided by value of R
- mean value of ST interval
- std of ST
- mode of ST
- mean of PQ divided by mean of ST
- skew of PT
- kurtosis of PT

Here positions of PQRST defined as:
- PQ = 0.16
- QRS = 0.1
- ST = 0.44
- R positions are detected by [P&T algorithm](http://www.robots.ox.ac.uk/~gari/teaching/cdt/A3/readings/ECG/Pan+Tompkins.pdf)
- Q = R - QRS / 2
- S = R + QRS / 2
- P = Q - PQ (start of P wave)
- T = S + ST (end of T wave)

If the value cannot be computed, it is replaced with 0.