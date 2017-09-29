from AF.parsers import ecg_recording_parser_af
from matplotlib import pyplot as plt
from os import path
from matplotlib import pyplot as plt
import numpy as np
import pprint, pickle
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser
from AF.simple_medical_analysers import wavelet_analysis
import wfdb
from os import path
from AF.parsers import ecg_recording_parser_af, ecg_recording_parser_ptb

def downsample(signal, modulo):

    new_signal = []
    for index, sample in enumerate(signal):
        if index % modulo == True:
            new_signal.append(sample)
    return new_signal


def get_list_of_intervals(signals_af):
    """Getting RR signals and returning list of intervals"""

    # Wavelet filtration
    numbers_of_signals = len(signals_af[0, :])
    numbers_of_samples = len(signals_af[:, 0])

    for i in range(numbers_of_signals):
        if numbers_of_samples % 2 == 1:
            signals_af = signals_af[:numbers_of_samples - 1, :]
        signals_af[:, i] = wavelet_analysis.filter_signal(signals_af[:, i], wavelet="db6", highcut=False)

    # Finding R-waves
    analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
    combined_rr = analyzerQRS.compare(signals_af, record_file=file_path, plotting=False, qrs_reading=False)

    # Creating list of isoelectrics lines
    rr_ia = RRIntervalsAnalyser.RRIntervalsAnalyser(analyzerQRS)
    list_of_intervals_chann0, rr_distances_chann0 = rr_ia.get_intervals(combined_rr, channel_no=0,
                                                                        time_margin=0.1)
    list_of_intervals_chann1, rr_distances_chann1 = rr_ia.get_intervals(combined_rr, channel_no=1,
                                                                        time_margin=0.1)

    return (list_of_intervals_chann0, list_of_intervals_chann1)

file_with_all_names = path.join("downloads", "ptb", "patients.txt")
file_object = open(file_with_all_names, "r")

for file_no in file_object:
    # if file_no < "07162":
    #     continue

    dataset_per_file_no = {
                "channel0": [],
               "channel1": []}


    try:
        file_no = file_no.replace("\n", "")
        file_path = path.join("downloads", "ptb", file_no)

        parser = ecg_recording_parser_ptb.ECGRecordingPTBDataParser()
        signal_0 = parser.parse(file_path + ".dat", channel_no=6, from_sample=0, to_sample=3*12*2000*4, modulo=12)
        signal_0_resampled = downsample(signal_0, modulo=4)

        signal_1 = parser.parse(file_path + ".dat", channel_no=7, from_sample=0, to_sample=3*12*2000*4, modulo=12)
        signal_1_resampled = downsample(signal_1, modulo=4)

        signals_ptb = np.empty((len(signal_0_resampled),2))

        signals_ptb[:,0] = signal_0_resampled
        signals_ptb[:,1] = signal_1_resampled


        (list_of_intervals_chann0, list_of_intervals_chann1) = get_list_of_intervals(signals_ptb)

        # Uptadting dataset
        if len(list_of_intervals_chann1) == len(list_of_intervals_chann0):
        # Saving isoelectric channel
            dataset_per_file_no["channel0"] += list_of_intervals_chann0
            dataset_per_file_no["channel1"] += list_of_intervals_chann1
        else:
            print("SUMA KONTROLNA SIĘ NIE ZGADZA!!!!!!! :(((( ")


        #
        # # Finding R-waves
        # analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
        # combined_rr = analyzerQRS.compare(signals_ptb, record_file=file_path, plotting=False, qrs_reading=False)
        #
        # # Creating list of isoelectrics lines
        # rr_ia = RRIntervalsAnalyser.RRIntervalsAnalyser(analyzerQRS)
        # list_of_intervals_chann0, rr_distances_chann0 = rr_ia.get_intervals(combined_rr, channel_no=0,
        #                                                                     time_margin=0.1)
        # list_of_intervals_chann1, rr_distances_chann1 = rr_ia.get_intervals(combined_rr, channel_no=1,
        #                                                                     time_margin=0.1)
        # print(len(signal_0))
        # print(len(signal_0_resampled))
        # plt.figure(1).clear()
        # plt.ion()
        # plt.subplot(2,2,1)
        # plt.plot(signal_0)
        # plt.subplot(2,2,2)
        # plt.plot(signal_0_resampled)


        plt.pause(0.05)

    except FileNotFoundError:
        print("Brak kompletu plików dla rekordu o nr: " + str(file_no))

    (token1, token2) = file_no.split("/")
    output = open('database/norm_data/' + token1 + "_" + token2 + '.pkl', 'wb')
    pickle.dump(dataset_per_file_no, output, -1)
    output.close()
