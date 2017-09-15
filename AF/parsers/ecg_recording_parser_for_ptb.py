from os import path
from matplotlib import pyplot as plt
import struct

# This class is parsing ECG recording from given sample to given sample

class ECGRecordingPTBDataParser(object):

    def __init__(self):
        pass

    def parse(self, dat_file_name, channel_no, modulo, from_sample, to_sample):
        return self._read_samples(dat_file_name, channel_no, modulo, from_sample, to_sample)

    def _read_samples(self, dat_file_name, channel_no, modulo, from_sample, to_sample):
        from_byte, to_byte = from_sample, to_sample
        samples = []
        with open(dat_file_name, "rb") as f:
            f.seek(from_byte)
            index_of_the_sample = 0

            for i in range(0, to_sample - from_sample + 1):
                sample_low = f.read(1)
                if not sample_low:
                    break

                sample_high = f.read(1)
                if not sample_high:
                    break

                if index_of_the_sample % modulo == channel_no:

                    value_low = ord(sample_low)
                    value_high = ord(sample_high)

                    if value_high > 127 :
                        value_high = 256 - ord(sample_high)
                        value_low = 256 - ord(sample_low)
                        value = -(value_high*256 + value_low + 1 )
                    else:
                        value = value_high*256 + value_low

                    samples.append(value)

                index_of_the_sample += 1

        return samples



