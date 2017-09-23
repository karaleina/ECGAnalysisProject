import numpy as np
import pywt


def calculate_wavelet_coeff_energy(coeff):

    # for normalization 1
    # number_of_elements_in_coeff = len(coeff)

    return sum(np.power(coeff, 2)) #/ number_of_elements_in_coeff


def get_all_normed_coeffs_energies(old_signal, coeffs_list):

    # for normalization
    old_signal_energy = calculate_wavelet_coeff_energy(old_signal)

    list_of_coeff_energy = []

    for coeffs in coeffs_list:
        inner_list=[]
        for coeff in coeffs:
            inner_list.append(calculate_wavelet_coeff_energy(coeff)/old_signal_energy)
        list_of_coeff_energy.append(inner_list)

    return list_of_coeff_energy


def filter_signal(signal, wavelet="db6", highcut=False):

    try:
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

        if highcut==True:
            cD = np.zeros(len(cA))
            cD2 = np.zeros(len(cD2))
            #cD3 = np.zeros(len(cD3))

        list_of_coeffs = [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD]
        return pywt.waverec(list_of_coeffs, wavelet)

    except ValueError:
        print("Wavelet transform except even number of samples only.")


def get_AF_coeffs_and_AF_signal(signal, wavelet="dmey"):

    old_signal = signal

    (cA, cD) = pywt.dwt(signal, wavelet)
    (cA2, cD2) = pywt.dwt(cA, wavelet)
    (cA3, cD3) = pywt.dwt(cA2, wavelet)
    (cA4, cD4) = pywt.dwt(cA3, wavelet)
    (cA5, cD5) = pywt.dwt(cA4, wavelet)
    (cA6, cD6) = pywt.dwt(cA5, wavelet)
    (cA7, cD7) = pywt.dwt(cA6, wavelet)
    (cA62, cD62 ) = pywt.dwt(cD5, wavelet)

    list_of_new_coeffs = [cA7, cD7, cD6, cD5]

    # Zerowanie niepotrzebnych
    old_coeffs = [cD4, cD3, cD2, cD]
    for one_coeff in old_coeffs:
        list_of_new_coeffs.append(np.zeros(len(one_coeff)))

    reconstructed_signal = pywt.waverec(list_of_new_coeffs, wavelet)

    return ([cD7, cD6, cA62], reconstructed_signal, old_signal) # do 3,5 Hz, do 7 Hz, do 11,5 Hz

