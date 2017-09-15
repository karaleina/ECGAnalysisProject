from AF.model import wfdbtools
from AF.parsers import  ecg_recording_data_parser
from scipy import signal
import path
import numpy as np
import sys


sys.path.append('home/karolina/PycharmProjects/atrialFibrillationAnalysisProject')

class Record():

    def __init__(self, path, database="MIT-AF", frequency="250"):

        self._path = path
        self._database = database
        self._frequency = frequency

        self._signals = None

    def get_path(self):
        return self._path

    def get_signal(self, start=0, interval=10, length_of_other_signal=1000):

        end = int(start + interval*1/self.get_frequency())
        parser = ecg_recording_data_parser.ECGRecordingDataParser()
        signals = np.array(parser.parse(str(self._path) + ".dat", start, end))

        #wfdbtools.rdsamp(record=self._path, start=start, interval=interval)

        print(signals)
        if self._frequency != 250:


            resampled1 = signal.resample(signals[:,0], num=length_of_other_signal)
            resampled2 = signal.resample(signals[:,1], num=length_of_other_signal)
            result = np.zeros((len(resampled1), 2))
            result[:,0] = resampled1
            result[:,1] = resampled2
            return result

        return signals

    def get_frequency(self):
        return self._frequency

    def set_frequency(self, frequency):
        self._frequency = frequency

if __name__ == "main":
    pass
