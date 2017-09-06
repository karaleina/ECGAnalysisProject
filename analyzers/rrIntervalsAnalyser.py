from matplotlib import pyplot as plt
from medical_objects import rrInterval
import numpy as np

class rrIntervalsAnalyser(object):

    def __init__(self, bothChannelsQRSDetector):

        self._sampling_ratio = bothChannelsQRSDetector._sampling_ratio
        self._signals = bothChannelsQRSDetector._signals

    def get_intervals(self, r_waves, channel_no, time_margin=0):

        samples_margin = time_margin * self._sampling_ratio
        channel = self._signals[:, channel_no]
        prev_r_wave_index = -1

        list_of_intervals = []
        rr_distanses = []

        for r_wave_index in r_waves:
            if prev_r_wave_index > 0:
                current_rr_interval = channel[int(prev_r_wave_index - samples_margin):int(r_wave_index + samples_margin)]
                list_of_intervals.append(rrInterval.RRInterval(current_rr_interval))
                rr_distanses.append(int(r_wave_index - prev_r_wave_index))

            prev_r_wave_index = r_wave_index

        #self.plot_histogram(rr_distanses, "Histogram odstępów RR")

        return list_of_intervals, rr_distanses

    def plot_histogram(self, signal, title):

        raw_signal = signal
        signal = np.multiply(signal, 1/self._sampling_ratio)

        plt.figure(1)
        plt.subplot(2,1,1)
        plt.title(title)

        plt.hist(raw_signal)
        plt.xlabel("[n]")
        plt.ylabel("Liczba zliczeń")

        plt.subplot(2,1,2)

        plt.hist(signal)
        plt.xlabel("[s]")
        plt.ylabel("Liczba zliczeń")

        plt.show()
