import numpy as np
import pywt


def filter_signal(signal, wavelet="db6"):

    try:
        (cA, cD) = pywt.dwt(signal, wavelet)
        (cA2, cD2) = pywt.dwt(cA, wavelet)
        (cA3, cD3) = pywt.dwt(cA2, wavelet)
        (cA4, cD4) = pywt.dwt(cA3, wavelet)
        (cA5, cD5) = pywt.dwt(cA4, wavelet)
        (cA6, cD6) = pywt.dwt(cA5, wavelet)
        (cA7, cD7) = pywt.dwt(cA6, wavelet)

        cA7 = np.zeros(len(cA7)) # odcięcie składowej stałej poniżej 1 Hz
        list_of_coeffs = [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD]

        return pywt.waverec(list_of_coeffs, wavelet)

    except ValueError:
        print("Wavelet transform except even number of samples only.")


def get_AF_energy(signal, wavelet="dmey"):

    old_signal = signal

    (cA, cD) = pywt.dwt(signal, wavelet)
    (cA2, cD2) = pywt.dwt(cA, wavelet)
    (cA3, cD3) = pywt.dwt(cA2, wavelet)
    (cA4, cD4) = pywt.dwt(cA3, wavelet)
    (cA5, cD5) = pywt.dwt(cA4, wavelet)
    (cA6, cD6) = pywt.dwt(cA5, wavelet)
    (cA52, cD52 ) = pywt.dwt(cD4, wavelet)

    #module = abs(np.max(signal) - np.min(signal))
    cALL = [list(cD6) + list(cD5) + list(cA52)]
    coeffs_energy = np.sum(np.power(cALL, 2))
    signal_energy = np.sum(np.power(old_signal, 2))
    norm_coeffs_energy = coeffs_energy /signal_energy
    return norm_coeffs_energy


