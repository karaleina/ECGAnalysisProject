import numpy as np
from AF.algorithms import r_wave_detection


class InvertionECGAnalyser(object):

     def __init__(self):
        pass

     @staticmethod
     def is_ecg_inverted( ecg_signal):

        # R_waves
        r_wave_detector = r_wave_detection.Thompkins()
        r_waves = r_wave_detector.detect_r_waves(ecg_signal)

        mean_R_value = sum(ecg_signal[r_waves])/len(r_waves)

        if mean_R_value > 0 :
            return False
        else:
            return True

     @staticmethod
     def automatically_invert_signals(signals):

         new_signals = signals.copy()

         for signal_index in range(len(signals[0, :])):
             current_signal = signals[:, signal_index]
             if InvertionECGAnalyser.is_ecg_inverted(current_signal):
                new_signals[:, signal_index] = np.multiply(current_signal,-1)

         # SIGNAL V2 ECG IS INVERTED
         new_signals[:, 0] = np.multiply(new_signals[:, 0], -1)

         return new_signals

     @staticmethod
     def manually_invert_signal(signal):
         return np.multiply(signal, -1)

