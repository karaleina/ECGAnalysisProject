from parsers import ecg_recording_data_parser
from medical import qrs_detector, qrs_compare, detection_combiner
from medical import get_atr

from os import path
from matplotlib import pyplot as plt
import numpy as np


class BothChannelsQRSAnalyser(object):

    def __init__(self):

        self._channel1_r_waves = []
        self._channel2_r_waves = []
        self._combined_rr = []
        self._reference = []

        # TODO Stworzyć obiekt record i tam to trzymać
        self._sampling_ratio = None
        self._tol_time = None

    def analyse(self, record, start_sample=0, stop_sample=10000, sampling_ratio=250, tol_time=0.05):

        self._sampling_ratio = sampling_ratio
        self._tol_time = tol_time

        parser = ecg_recording_data_parser.ECGRecordingDataParser()
        signals = np.array(parser.parse(str(record) + ".dat", start_sample, stop_sample))
        self._reference = np.array(get_atr.get_true_r_waves(record, start_sample, stop_sample))
        print(self._reference)

        panThompikns = qrs_detector.QRSPanThompkinsDetector()
        self._channel1_r_waves = np.array(panThompikns.detect_qrs(signals[:,0]))[:,0]
        self._channel2_r_waves = np.array(panThompikns.detect_qrs(signals[:,1]))[:,0]


        dc = detection_combiner.DetectionCombiner()
        self._combined_rr = dc.combine(channel1=self._channel1_r_waves, channel2=self._channel2_r_waves,
                                       sampling_ratio=self._sampling_ratio, tol_time=self._tol_time)

        self.calculate_and_print_sensitivity_and_specifity()

        self.plot(1, signals[:, 0], "Sygnał z kanału 0")
        self.plot(2, signals[:, 1], "Sygnał z kanału 1")
        plt.show()

    def calculate_and_print_sensitivity_and_specifity(self):

        compareQRS = qrs_compare.QRSCompare()
        sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._channel1_r_waves,
                                                               sampling_ratio=self._sampling_ratio, tol_time=self._tol_time)
        print("Czułość i specyficzność na podstwie pojedynczego kanału nr " + str(1) + ":" + str(sensivity), str(specifity))
        sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._channel2_r_waves,
                                                               sampling_ratio=self._sampling_ratio, tol_time=self._tol_time)
        print("Czułość i specyficzność na podstwie pojedynczego kanału nr " + str(2) + ":" + str(sensivity),
              str(specifity))
        sensivity, specifity = compareQRS.compare_segmentation(reference=self._reference, test=self._combined_rr,
                                                               sampling_ratio=self._sampling_ratio, tol_time=self._tol_time)
        print('Czułość i specyficzność na podstawie obydwu kanałow: ', str(sensivity), str(specifity))

    def rr_intervals_analyse(self, r_waves, channel):

        # TODO

        pass



    def plot(self, idx, signal, title):

        # Plotting
        plt.figure(idx)
        plt.plot(signal)
        plt.plot(self._combined_rr, np.ones_like(self._combined_rr), "m*", label="Wyznaczone załamki")
        plt.title(title)
        plt.plot(self._reference, 40 + np.ones_like(self._reference), "b*", label="Referencyjne załamki")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)




record = path.join("downloads", "04126")

analyzer = BothChannelsQRSAnalyser()
analyzer.analyse(record, start_sample=0, stop_sample=10000)