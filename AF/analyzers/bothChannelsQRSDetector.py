import numpy as np
from AF.algorithms import r_wave_detection
from matplotlib import pyplot as plt
from AF.parsers import ecg_recording_parser_af
from AF.simple_medical_analysers import qrs_compare, detection_combiner, wavelet_analysis

from AF.simple_medical_analysers import get_atr


class BothChannelsQRSDetector(object):

    def __init__(self):

        self._signals = []
        self._reference = []
        self._number_of_signals = None
        self._record_file = None
        self._start_sample = None
        self._stop_sample = None

        self._combined_rr = []

        # TODO Stworzyć obiekt record i tam to trzymać
        self._sampling_ratio = None
        self._tol_compare_time = None

    def compare(self, signals, sampling_ratio=250, margin_r_waves_time=0.1, start_sample=0, stop_sample=1000000000000,
                record_file=None, info=True, plotting=True, qrs_reading=True):

        """Signals are 2D numpy array where:
            2. index (rows) stands for a channel number
            1. index (columns) stands for samples
        """

        self._signals = signals
        self._number_of_signals = len(signals[0,:])
        self._sampling_ratio = sampling_ratio
        self._tol_compare_time = margin_r_waves_time
        self._record_file = record_file
        self._start_sample = start_sample
        self._stop_sample = stop_sample

        detector = r_wave_detection.Hamilton()#QRSPanThompkinsDetector()
        dc = detection_combiner.DetectionCombiner()

        if qrs_reading == True:
            self._reference = np.array(get_atr.get_true_r_waves(self._record_file, self._start_sample, self._stop_sample))

        old_channel_r_waves = []

        for signal_index in range(self._number_of_signals):
            r_waves = np.array(detector.detect_r_waves(self._signals[:, signal_index])).ravel()
            new_channel_r_waves = dc.verify(r_waves, sampling_ratio=self._sampling_ratio,
                                         tol_compare_time=self._tol_compare_time)
            self._combined_rr = dc.combine(old_channel_r_waves, new_channel_r_waves)
            old_channel_r_waves = new_channel_r_waves

        if info == True:
            # TODO printing for multiple channels
            self.print_combining_results()

        if plotting == True:
            for signal_index in range(self._number_of_signals):
                self.plot_my_plot(signal_index, self._signals[:, signal_index], self._combined_rr)
            plt.show()

        return self._combined_rr

    def print_combining_results(self):

        # TODO for multiple channels
        pass

        # compareQRS = qrs_compare.QRSCompare()
        # sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._channel1_r_waves,
        #                                                        sampling_ratio=self._sampling_ratio, tol_time=self._tol_compare_time)
        # print("Czułość i specyficzność na podstwie pojedynczego kanału nr " + str(1) + ":" + str(sensivity), str(specifity))
        # sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._channel2_r_waves,
        #                                                        sampling_ratio=self._sampling_ratio, tol_time=self._tol_compare_time)
        # print("Czułość i specyficzność na podstwie pojedynczego kanału nr " + str(2) + ":" + str(sensivity),
        #       str(specifity))
        # sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._combined_rr,
        #                                                        sampling_ratio=self._sampling_ratio, tol_time=self._tol_compare_time)
        # print('Czułość i specyficzność na podstawie obydwu kanałow: ', str(sensivity), str(specifity))
        # print('Liczba załamków referencyjnych/znalezionych: ' + str(len(self._reference)) + " / " + str(len(self._combined_rr)))

    def plot_my_plot(self, idx, signal, rr):
        plt.figure(idx)
        plt.plot(signal)
        plt.plot(rr, np.ones_like(rr), "m*", label="Wyznaczone załamki")
        # plt.title(title)
        # plt.plot(self._reference, 40 + np.ones_like(self._reference), "b*", label="Referencyjne załamki")
        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #                    ncol=2, mode="expand", borderaxespad=0.)
