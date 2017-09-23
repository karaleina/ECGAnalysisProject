from os import path
from matplotlib import pyplot as plt
import numpy as np
import pprint, pickle

from AF.parsers import ecg_recording_parser_af
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser
from AF.simple_medical_analysers import wavelet_analysis

# Parsing signal
number_af = "04746"
file_path = path.join("downloads", "af", number_af)

start_samples = [279600, 125750]
end_samples = [299200, 301200]

dataset = {"channel0" : [],
           "channel1" : []}

for i in range(len(start_samples)):

    start_sample = start_samples[i] - 50
    end = end_samples[i] + 50


    parser = ecg_recording_parser_af.ECGRecordingDataParser()
    signals_af = np.array(parser.parse(dat_file_name=file_path + ".dat", from_sample=start_sample, to_sample=end))

    # Wavelet filtration
    numbers_of_signals = len(signals_af[0,:])
    numbers_of_samples = len(signals_af[:,0])

    for i in range(numbers_of_signals):
        if numbers_of_samples % 2 == 1:
            signals_af = signals_af[:numbers_of_samples-1,:]
        signals_af[:,i] = wavelet_analysis.filter_signal(signals_af[:,i], wavelet="db6", highcut=False)

    # Finding R-waves
    analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
    combined_rr = analyzerQRS.compare(signals_af, record_file=file_path)

    # Creating list of isoelectrics lines
    rr_ia = RRIntervalsAnalyser.RRIntervalsAnalyser(analyzerQRS)
    list_of_intervals_chann0, rr_distances_chann0 = rr_ia.get_intervals(combined_rr, channel_no=0, time_margin=0.1)
    list_of_intervals_chann1, rr_distances_chann1 = rr_ia.get_intervals(combined_rr, channel_no=1, time_margin=0.1)

    list_of_isoelectrics_chann0 = rr_ia.get_list_of_intervals_isoelectric_line(list_of_intervals_chann0, interval_samples=100)
    list_of_isoelectrics_chann1 = rr_ia.get_list_of_intervals_isoelectric_line(list_of_intervals_chann1, interval_samples=100)

    #
    # # Plotting
    # for index, interval_chann0 in enumerate(list_of_intervals_chann0):
    #
    #     interval_signal_chann0 = interval_chann0.get_signal()
    #     interval_signal_chann1 = list_of_intervals_chann1[index].get_signal()
    #
    #     isoelectric_interval_chann0 = list_of_isoelectrics_chann0[index].get_signal()
    #     isoelectric_interval_chann1 = list_of_isoelectrics_chann1[index].get_signal()
    #
    #     # Plotting
    #     plt.ion()
    #     plt.figure(2).clear()
    #     plt.subplot(2, 1, 1)
    #     plt.plot(interval_signal_chann0)
    #     plt.plot(interval_signal_chann1)
    #     plt.ylabel("EKG")
    #
    #     plt.subplot(2, 1, 2)
    #     plt.plot(isoelectric_interval_chann0)
    #     plt.plot(isoelectric_interval_chann1)
    #     plt.ylabel("Linia izoelektryczna")
    #
    #     plt.pause(2)
    # plt.show()

    # Saving

    if len(list_of_intervals_chann1) == len(list_of_intervals_chann0):
        #Saving isoelectric channel
        dataset["channel0"] += list_of_intervals_chann0
        dataset["channel1"] += list_of_intervals_chann1
    else:
        print("SUMA KONTROLNA SIĘ NIE ZGADZA!!!!!!!")

output = open('database/af_data/' + number_af + '.pkl', 'wb')
pickle.dump(dataset, output, -1)
output.close()



