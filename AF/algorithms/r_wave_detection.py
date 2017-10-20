import numpy as np
from biosppy import ecg

from AF.algorithms.qrs_pan_thomkins_detector import QRSPanThompkinsDetector

#
# #Dziedziczenie???
# class Algorithm(object):
#     def name(self):
#         raise NotImplementedError()
#     def detect_r_waves(self, channel):
#         raise NotImplementedError()


class Christov(object):
    def name(self):
        return "Christov"

    def detect_r_waves(self, channel):
        return np.array((ecg.christov_segmenter(channel, sampling_rate=250)))[0]

class Ssf(object):
    def name(self):
        return "SSF"

    def detect_r_waves(self, channel):
        return np.array((ecg.ssf_segmenter(channel, sampling_rate=250)))[0]

class Gamboa(object):
    def name(self):
        return "Gamboa"

    def detect_r_waves(self, channel):
        return np.array(ecg.gamboa_segmenter(np.array(channel), sampling_rate=250))[0]

class Thompkins(object):
    def name(self):
        return "Pan-Thompkins"

    def detect_r_waves(self, channel):

        detector = QRSPanThompkinsDetector()
        a = detector.detect_qrs(channel)
        b = a

        r_waves_thompkins_chann = []
        for element in b:
            r_waves_thompkins_chann.append(element[0])
        return r_waves_thompkins_chann

class Engzee(object):
    def name(self):
        return "Engzee"

    def detect_r_waves(self, channel):
        return np.array(ecg.hamilton_segmenter(np.array(channel), sampling_rate=250))[0]

class Hamilton(object):
    def name(self):
        return "Hamilton"

    def detect_r_waves(self, channel):
        return np.array(ecg.hamilton_segmenter(np.array(channel), sampling_rate=250))[0]



algorithms = [Christov(), Thompkins(), Ssf(), Engzee(), Hamilton(), Gamboa()]
