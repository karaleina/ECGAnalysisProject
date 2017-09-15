from AF.parsers import  ecg_recording_parser_af, ecg_recording_parser_ptb

import numpy as np

class Record():

    def __init__(self, path, database=None):

        self._path = path
        self._database = database
        self._frequency = None
        self._signals = None

        if self._database == "af":
            self._frequency = 250 # Hz
        elif self._database == "ptb":
            self._frequency = 1000 # Hz
        else:
            print("Nieznana częstotliwość próbkowania!")

    def get_path(self):
        return self._path

    def get_signals(self, start=0, interval=10):
        """Returns 2D array :
        1) dimension: channel_idxes
        2) dimension: sample_idxes"""

        end = int(start + interval*self.get_frequency())
        self._signals = np.zeros((end - start + 1, 2))

        if self._database == "ptb":
            parser = ecg_recording_parser_ptb.ECGRecordingPTBDataParser()
            for signal_index in [6, 7]:
                self._signals[:, signal_index-6] = parser.parse(self._path + ".dat", channel_no=signal_index,
                                            modulo=12, from_sample=start, to_sample=12*(end+1), invert=False)
        elif self._database == "af":
            parser = ecg_recording_parser_af.ECGRecordingDataParser()
            self._signals[:,:] = parser.parse(self._path + ".dat", from_sample=start, to_sample=end)
        else:
            print("Brak zdefiniowanego rodzaju danych do parsowania")

        return (self._signals)

    def get_frequency(self):
        return self._frequency

    def set_frequency(self, frequency):
        self._frequency = frequency

if __name__ == "main":
    pass
