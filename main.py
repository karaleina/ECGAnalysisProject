from parsers import ecg_recording_data_parser
from medical import qrs_detector, qrs_compare
from tools.wfdbtools import rdann

from matplotlib import pyplot as plt
from os import path
import numpy as np


def get_true_r_waves(path_string, start_sample, stop_sample):

    result_qrs = rdann(path_string, 'qrs', start=0, end=stop_sample, types=[])
    r_waves = []

    for element in result_qrs:
        if start_sample <= element[0] <= stop_sample:
            r_waves.append(element[0])
        else:
            break

    return r_waves


# Variables

start_sample = 0
stop_sample = 10000

dictionary = "downloads"
record = path.join(dictionary, "04126")

# Parsing EKG

parser = ecg_recording_data_parser.ECGRecordingDataParser()
signals = parser.parse(str(record)+".dat", start_sample, stop_sample)
signals = np.array(signals)

panThompikns = qrs_detector.QRSPanThompkinsDetector()
compareQRS = qrs_compare.QRSCompare()

for i in [0, 1]:
    # R waves calculating
    # TODO Uwzględnianie obydwu odprowadzeń przy podawaniu wyniku
    r_waves = np.array(panThompikns.detect_qrs(signals[:,i]))

    # R reference reading from the file
    wfdb_r_waves = get_true_r_waves(record, start_sample, stop_sample)

    # Plotting
    plt.figure(1)
    plt.title("Załamki R znalezione przez algorytm")
    plt.subplot(2, 1, i + 1)
    plt.plot(signals[:,i])
    plt.xlabel("[n]")
    plt.ylabel("EKG")
    plt.plot(r_waves[:,0], r_waves[:,1], "m*")
    plt.plot(wfdb_r_waves, np.ones_like(wfdb_r_waves), "g*")


    plt.figure(2)
    plt.title("Załamki R referencyjne")
    plt.subplot(2, 1, i + 1)
    plt.plot(signals[:,i])
    plt.xlabel("[n]")
    plt.ylabel("EKG")
    plt.plot(wfdb_r_waves, np.ones_like(wfdb_r_waves), "g*")

    sensivity, specifity = compareQRS.compare_segmentation(reference=wfdb_r_waves, test=r_waves[:,0], tol_time=0.1)
    print(sensivity, specifity)

plt.show()



