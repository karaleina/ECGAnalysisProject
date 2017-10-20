from os import path
from matplotlib import pyplot as plt
import wfdb
import pywt
import numpy as np
import scipy.signal


def calculate_number_of_samples(time_of_record, fs):
    return int(fs * time_of_record)


def antyaliasing_filtration_from_1000_to_128(signal, wavelet = "db6"):

    # Nyquist: 1000/2 = 500Hz -> Aim: 128/2 = 64Hz

    (cA, cD) = pywt.dwt(signal, wavelet) # 250 Hz
    (cA2, cD2) = pywt.dwt(cA, wavelet) # 125 Hz
    (cA3, cD3) = pywt.dwt(cA2, wavelet) # 63 Hz

    cD = np.zeros(len(cD))
    cD2 = np.zeros(len(cD2))
    cD3 = np.zeros(len(cD3))

    list_of_coeffs = [cA3, cD3, cD2, cD]

    return pywt.waverec(list_of_coeffs, wavelet)


time_of_record = 15.625 #s


# Dane z AFTDB:
print("Liczba próbek do wczytania dla AFTDB", calculate_number_of_samples(time_of_record, fs=128))
with open("downloads/af_term/aftdb_record_names") as patients_list:
    for patient_id in patients_list:
        patient_id = patient_id.replace("\n", "")
        subdir, record_no = patient_id.split("/")
        if subdir == "learning-set":
            filepath = path.join("downloads", "af_term", subdir, record_no)
            record = wfdb.rdsamp(filepath)
            n = calculate_number_of_samples(time_of_record, fs=128)
            signals, fields = wfdb.srdsamp(filepath, sampfrom= record.siglen - n - 1, sampto= record.siglen - 1, channels=None, pbdir=None)
            channel0 = signals[:,0]
            channel1 = signals[:,1]

            # TODO RR

            # TODO SAVING records

# Dane z PTB:
print("Liczba próbek do wczytania dla PTB", calculate_number_of_samples(time_of_record, fs=1000))
with open("downloads/ptb/patients.txt") as patients_list:
    for patient_id in patients_list:
        patient_id = patient_id.replace("\n", "")
        subdir, record_no = patient_id.split("/")

        filepath = path.join("downloads", "ptb", subdir, record_no)
        record = wfdb.rdsamp(filepath)
        n = calculate_number_of_samples(time_of_record, fs=1000)
        signals, fields = wfdb.srdsamp(filepath, sampfrom= record.siglen - n - 1, sampto= record.siglen - 1, channels=None, pbdir=None)

        channel0 = signals[:, 6]
        channel1 = signals[:, 7]

        # ANTYALIASING FILTRATION
        channel0_filtered = antyaliasing_filtration_from_1000_to_128(channel0)
        channel1_filtered = antyaliasing_filtration_from_1000_to_128(channel1)

        # DOWNSAMPLING
        channel0_downsampled = scipy.signal.resample(channel0_filtered, calculate_number_of_samples(time_of_record, fs=128))
        channel1_downsampled = scipy.signal.resample(channel1_filtered, calculate_number_of_samples(time_of_record, fs=128))

        # TODO RR

        # TODO SAVING records
