import numpy as np
import pywt


def calculate_wavelet_coeff_energy(coeff):

    return sum(np.power(coeff, 2))


def filter_signal(signal, wavelet="db6"):

    (cA, cD) = pywt.dwt(signal, wavelet)
    (cA2, cD2) = pywt.dwt(cA, wavelet)
    (cA3, cD3) = pywt.dwt(cA2, wavelet)
    (cA4, cD4) = pywt.dwt(cA3, wavelet)
    (cA5, cD5) = pywt.dwt(cA4, wavelet)
    (cA6, cD6) = pywt.dwt(cA5, wavelet)
    (cA7, cD7) = pywt.dwt(cA6, wavelet)

    cA7 = np.zeros(len(cA7))
    # cD = np.zeros(len(cD))
    # cD2 = np.zeros(len(cD2))

    list_of_coeffs = [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD]
    return pywt.waverec(list_of_coeffs, wavelet)