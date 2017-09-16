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

    def get_signals(self, start=0, interval=10, number_of_signals=2):
        """Returns 2D array :
        1) dimension: channel_idxes
        2) dimension: sample_idxes"""

        end = int(start + interval*self.get_frequency())

        self._signals = np.zeros((end - start + 1, number_of_signals))

        if self._database == "ptb":
            parser = ecg_recording_parser_ptb.ECGRecordingPTBDataParser()
            index = 0
            if number_of_signals!=2:
                signal_indexes = range(0,number_of_signals)
            else:
                signal_indexes = [6, 7]

            for signal_index in signal_indexes:
                print(signal_index)
                self._signals[:, index] = parser.parse(self._path + ".dat", channel_no=signal_index,
                                            modulo=12, from_sample=start, to_sample=12*(end+1)-1, invert=False)
                index +=1
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
