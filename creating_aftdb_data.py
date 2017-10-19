import pprint, pickle
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser
from AF.simple_medical_analysers import wavelet_analysis
from os import path
from matplotlib import pyplot as plt
from AF.parsers import ecg_recording_parser_af
import wfdb
import numpy as np


def get_list_of_intervals(signals_af, filepath):
    """Getting signals and returning list of intervals"""

    # Wavelet filtration
    numbers_of_signals = len(signals_af[0, :])
    numbers_of_samples = len(signals_af[:, 0])

    for i in range(numbers_of_signals):
        if numbers_of_samples % 2 == 1:
            signals_af = signals_af[:numbers_of_samples - 1, :]
        signals_af[:, i] = wavelet_analysis.filter_signal(signals_af[:, i], wavelet="db6")

    # Finding R-waves
    analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
    combined_rr = analyzerQRS.compare(signals_af, record_file=filepath, plotting=False, qrs_reading=False)

    # Creating list of isoelectrics lines
    rr_ia = RRIntervalsAnalyser.RRIntervalsAnalyser(analyzerQRS)
    list_of_intervals_chann0, rr_distances_chann0 = rr_ia.get_intervals(combined_rr, channel_no=0,
                                                                        time_margin=0.1)
    list_of_intervals_chann1, rr_distances_chann1 = rr_ia.get_intervals(combined_rr, channel_no=1,
                                                                        time_margin=0.1)
    return (list_of_intervals_chann0, list_of_intervals_chann1)


dir = path.join("downloads", "af_term")
i = 0

dataset_aftdb = {}

with open("records_names") as patients_list:

    for patient_id in patients_list:

        patient_id = patient_id.replace("\n", "")
        subdir, record_no = patient_id.split("/")

        if subdir != "learning-set":
            pass
        else:
            dataset_aftdb[patient_id] = {"channel0": [],
                                         "channel1": []}

            filepath = path.join(dir, subdir, record_no)
            record = wfdb.rdsamp(filepath)

            signals, fields = wfdb.srdsamp(filepath, sampfrom= record.siglen - 2001, sampto= record.siglen - 1, channels=None, pbdir=None)
            filepath = "downloads/af_term" + "/" + subdir + "/" + record_no
            (list_of_intervals_chann0, list_of_intervals_chann1) = get_list_of_intervals(signals, filepath)

            # Uptadting dataset
            if len(list_of_intervals_chann1) == len(list_of_intervals_chann0):
                dataset_aftdb[patient_id]["channel0"] += list_of_intervals_chann0
                dataset_aftdb[patient_id]["channel1"] += list_of_intervals_chann1
            else:
                print("SUMA KONTROLNA SIĘ NIE ZGADZA!!!!!!!")

filepath_new_pickle = path.join("database", "af_term" + '.pkl')
output = open(filepath_new_pickle, 'wb')
pickle.dump(dataset_aftdb, output, -1)
output.close()

print(dataset_aftdb)
