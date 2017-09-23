from os import path
import cv2
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser, dimensionAnalyzer, invertionAnalyzer
from AF.simple_medical_analysers import wavelet_analysis
from AF.model import record
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal


name_str = "04015" #["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091", "05121", "08455"]):

interval = 20
# end = 20 * 250
# start = 85000
# list_of_first_indexes = [start]
#
# previous_element = start
# for i in range(100):
#     new_element = previous_element + end
#     list_of_first_indexes.append(new_element)
#     previous_element = new_element

for start in [31700]:#list_of_first_indexes:
    path_af = path.join("downloads", "af", name_str)
    record_atrial_fibrillation = record.Record(path_af, database="af")

    # PARAMS
    record_atrial_fibrillation.set_frequency(250)

    # SIGNALS
    signals_af = record_atrial_fibrillation.get_signals(start, interval)
    signal_0 = signals_af[:,0]
    signal_1 = signals_af[:,1]

    # QRS Analysis and RR Analysis
    analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
    combined_rr = analyzerQRS.compare(signals_af, record_file=path_af)

    rr_prev_index = 0
    index_sample = start

    for index, rr_index in enumerate(combined_rr):

        print(index)
        if index > 0:

            interval_signal_0 = signal_0[int(rr_prev_index):int(rr_index)]
            interval_signal_1 = signal_1[int(rr_prev_index):int(rr_index)]

            print("Index sample : " + str(index_sample))
            plt.ion()
            plt.figure(1).clear()

            plt.subplot(2,1,1)
            plt.plot(interval_signal_0)
            plt.title("Interval channel 1")

            plt.subplot(2,1,2)
            plt.plot(interval_signal_1)
            plt.title("Interval channel 2")

            try:
                number = "2200" + str(index_sample)
                plt.figure(2).clear()
                plt.subplot(2,1,1)
                plt.plot(signal_0[int(rr_prev_index - 200):int(rr_index + 200)])

                plt.subplot(2, 1, 2)
                plt.plot(signal_1[int(rr_prev_index - 200):int(rr_index + 200)])

            except IndexError:
                print("Index error")

            plt.pause(0.05)
            input("Press Enter to continue...")

        rr_prev_index = rr_index

        index_sample = start + rr_index

    input("Press Enter to continue...")
    plt.show()
