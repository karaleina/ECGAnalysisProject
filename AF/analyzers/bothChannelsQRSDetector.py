import numpy as np
from AF.algorithms import r_wave_detection
from matplotlib import pyplot as plt
from AF.parsers import ecg_recording_data_parser
from AF.simple_medical_analysers import qrs_compare, detection_combiner, wavelet_analysis

from AF.simple_medical_analysers import get_atr


class BothChannelsQRSDetector(object):

    def __init__(self):

        self._channel1_r_waves = []
        self._channel2_r_waves = []
        self._combined_rr = []
        self._reference = []
        self._signals = []

        # TODO Stworzyć obiekt record i tam to trzymać
        self._sampling_ratio = None
        self._tol_compare_time = None

    def analyse(self, record, start_sample=0, stop_sample=1000, sampling_ratio=250, margin_r_waves_time=0.1, info=True, plotting=True, qrs_reading=True):

        # len should always be % 2 == 0 for wavelet reasons
        if stop_sample - start_sample % 2 != 0:
            stop_sample += 1

        self._sampling_ratio = sampling_ratio
        self._tol_compare_time = margin_r_waves_time

        parser = ecg_recording_data_parser.ECGRecordingDataParser()
        self._signals = np.array(parser.parse(str(record) + ".dat", start_sample, stop_sample))

        # TODO Before Thompkins filtration
        self._signals[:, 0] = wavelet_analysis.filter_signal(self._signals[:, 0].ravel())
        self._signals[:, 1] = wavelet_analysis.filter_signal(self._signals[:, 1].ravel())

        # TODO Other detection algorithms
        # TODO Other kind of detection

        #panThompikns = qrs_pan_thomkins_detector.QRSPanThompkinsDetector()
        qrsDetector = r_wave_detection.Christov()
        self._channel1_r_waves = np.array(qrsDetector.detect_r_waves(self._signals[:,0])).ravel()
        self._channel2_r_waves = np.array(qrsDetector.detect_r_waves(self._signals[:,1])).ravel()

        print("--"*100)
        print(self._channel1_r_waves)
        print(self._channel2_r_waves)

        dc = detection_combiner.DetectionCombiner()
        self._channel1_r_waves = dc.verify(self._channel1_r_waves, sampling_ratio=self._sampling_ratio,
                                           tol_compare_time=self._tol_compare_time)
        self._channel2_r_waves = dc.verify(self._channel2_r_waves, sampling_ratio=self._sampling_ratio,
                                           tol_compare_time=self._tol_compare_time)
        self._combined_rr = dc.combine(channel1=self._channel1_r_waves, channel2=self._channel2_r_waves,
                                       sampling_ratio=self._sampling_ratio, tol_compare_time=self._tol_compare_time)

        if qrs_reading == True and info == True:
            self._reference = np.array(get_atr.get_true_r_waves(record, start_sample, stop_sample))
            self.print_combining_results()

        if plotting == True:
            self.plot_my_plot(1, self._signals[:, 0], self._channel1_r_waves, "Sygnał z kanału 0")
            self.plot_my_plot(2, self._signals[:, 1], self._channel2_r_waves, "Sygnał z kanału 1")
            plt.show()

        return self._combined_rr


    def print_combining_results(self):

        compareQRS = qrs_compare.QRSCompare()
        sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._channel1_r_waves,
                                                               sampling_ratio=self._sampling_ratio, tol_time=self._tol_compare_time)
        print("Czułość i specyficzność na podstwie pojedynczego kanału nr " + str(1) + ":" + str(sensivity), str(specifity))
        sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._channel2_r_waves,
                                                               sampling_ratio=self._sampling_ratio, tol_time=self._tol_compare_time)
        print("Czułość i specyficzność na podstwie pojedynczego kanału nr " + str(2) + ":" + str(sensivity),
              str(specifity))
        sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._combined_rr,
                                                               sampling_ratio=self._sampling_ratio, tol_time=self._tol_compare_time)
        print('Czułość i specyficzność na podstawie obydwu kanałow: ', str(sensivity), str(specifity))
        print('Liczba załamków referencyjnych/znalezionych: ' + str(len(self._reference)) + " / " + str(len(self._combined_rr)))


    def plot_my_plot(self, idx, signal, rr, title):

        plt.figure(idx)
        plt.plot(signal)
        plt.plot(rr, np.ones_like(rr), "m*", label="Wyznaczone załamki")
        plt.title(title)
        plt.plot(self._reference, 40 + np.ones_like(self._reference), "b*", label="Referencyjne załamki")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
