from os import path
from matplotlib import pyplot as plt
import wfdb
import pywt
import numpy as np
import pickle
import scipy.signal
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser


def calculate_number_of_samples(time_of_record, fs):
    return int(fs * time_of_record)


def save_with_pickle(data, pickle_file):
    # Pickle the 'data' dictionary using the highest protocol available.
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


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
aftdb_dataset = {}
ptb_dataset = {}

# Dane z AFTDB:
print("Liczba próbek do wczytania dla AFTDB", calculate_number_of_samples(time_of_record, fs=128))
with open("downloads/af_term/aftdb_record_names") as patients_list:
    for patient_id in patients_list:
        patient_id = patient_id.replace("\n", "")
        subdir_patientname, record_no = patient_id.split("/")

        if subdir_patientname == "learning-set":
            filepath = path.join("downloads", "af_term", subdir_patientname, record_no)
            record = wfdb.rdsamp(filepath)
            n = calculate_number_of_samples(time_of_record, fs=128)
            signals, fields = wfdb.srdsamp(filepath, sampfrom= record.siglen - n - 1, sampto= record.siglen - 1, channels=None, pbdir=None)

            # GETTING RR INTERVALS
            rr_detector = bothChannelsQRSDetector.BothChannelsQRSDetector()
            combined_rr = rr_detector.compare(signals=signals[:, 0:2], sampling_ratio=128)
            rr_ia = RRIntervalsAnalyser.RRIntervalsAnalyser(sampling_ratio=128, signals=signals[:,0:2])

            list_of_intervals_chann0, rr_distances_chann0 = rr_ia.get_intervals(combined_rr, channel_no=0,
                                                                                time_margin=0.1)
            list_of_intervals_chann1, rr_distances_chann1 = rr_ia.get_intervals(combined_rr, channel_no=1,
                                                                                time_margin=0.1)
            # SAVE TO DICT WITH PATIENT INFO
            patient_name = record_no
            data_for_patient = {"channel0": list_of_intervals_chann0,
                                "channel1": list_of_intervals_chann1}
            aftdb_dataset[patient_name] = data_for_patient

            if record_no == 'n03':

                plt.ioff()
                plt.figure(1).clear()
                plt.subplot(2, 1, 1)
                plt.plot(signals[:, 0])
                plt.subplot(2, 1, 2)
                plt.plot(signals[:, 1])
                plt.show()

                for index, interval in enumerate(list_of_intervals_chann0):
                    plt.ion()
                    plt.figure(1).clear()
                    plt.subplot(2,1,1)
                    plt.plot(interval.get_signal())
                    plt.subplot(2,1,2)
                    plt.plot(list_of_intervals_chann1[index].get_signal())
                    plt.pause(0.5)

        else:
            break

# Dane z PTB:
print("Liczba próbek do wczytania dla PTB", calculate_number_of_samples(time_of_record, fs=1000))
with open("downloads/ptb/patients.txt") as patients_list:
    for patient_id in patients_list:
        patient_id = patient_id.replace("\n", "")
        patient_name, record_no = patient_id.split("/")

        filepath = path.join("downloads", "ptb", patient_name, record_no)
        record = wfdb.rdsamp(filepath)
        n = calculate_number_of_samples(time_of_record, fs=1000)
        signals, fields = wfdb.srdsamp(filepath, sampfrom= record.siglen - n - 1, sampto= record.siglen - 1, channels=None, pbdir=None)

        channel0 = signals[:, 6]
        channel1 = signals[:, 7]

        # ANTYALIASING FILTRATION
        channel0_filtered = antyaliasing_filtration_from_1000_to_128(channel0)
        channel1_filtered = antyaliasing_filtration_from_1000_to_128(channel1)

        # DOWNSAMPLING
        af_number_of_samples = calculate_number_of_samples(time_of_record, fs=128)
        channel0_downsampled = scipy.signal.resample(channel0_filtered, af_number_of_samples)
        channel1_downsampled = scipy.signal.resample(channel1_filtered, af_number_of_samples)

        # DATASET
        dataset = np.array([[x, y] for x, y in zip(channel0_downsampled, channel1_downsampled)])

        # GETTING RR INTERVALS
        rr_detector = bothChannelsQRSDetector.BothChannelsQRSDetector()
        combined_rr = rr_detector.compare(signals=dataset, sampling_ratio=128)
        rr_ia = RRIntervalsAnalyser.RRIntervalsAnalyser(sampling_ratio=128, signals=dataset[:, 0:2])

        list_of_intervals_chann0, rr_distances_chann0 = rr_ia.get_intervals(combined_rr, channel_no=0,
                                                                            time_margin=0.1)
        list_of_intervals_chann1, rr_distances_chann1 = rr_ia.get_intervals(combined_rr, channel_no=1,
                                                                            time_margin=0.1)
        # for index, interval in enumerate(list_of_intervals_chann0):
        #     plt.ion()
        #     plt.figure(0).clear()
        #     plt.subplot(2,1,1)
        #     plt.plot(interval.get_signal())
        #     plt.subplot(2,1,2)
        #     plt.plot(list_of_intervals_chann1[index].get_signal())
        #     plt.pause(0.05)

        # TODO SAVING records

        # SAVE TO DICT WITH PATIENT INFO
        if patient_name in ptb_dataset:
            ptb_dataset[patient_name]["channel0"].append(list_of_intervals_chann0)
            ptb_dataset[patient_name]["channel1"].append(list_of_intervals_chann1)
        else:
            ptb_dataset[patient_name] = {"channel0": list_of_intervals_chann0,
                                         "channel1": list_of_intervals_chann1}

# SAVING
save_with_pickle(aftdb_dataset, "database/aftdb.pkl")
print(len(aftdb_dataset))

save_with_pickle(ptb_dataset, "database/ptb.pkl")
print(len(ptb_dataset))
