# This class is parsing ECG recording from given sample to given sample

class ECGRecordingDataParser(object):

    def __init__(self):
        pass

    def parse(self, dat_file_name, from_sample, to_sample):
        return self._read_samples(dat_file_name, from_sample, to_sample)

    def _read_samples(self, dat_file_name, from_sample, to_sample):
        from_byte, to_byte = self._sample_range_to_byte_range(from_sample, to_sample)
        samples = []
        with open(dat_file_name, "rb") as f:
            f.seek(from_byte)
            for i in range(0, to_sample - from_sample + 1):
                sample = f.read(3)
                if not sample:
                    break
                sample_bytes = [char for char in sample]
                first_value = self._complement_of_two_12_bit(sample_bytes[0] | ((sample_bytes[1] & 0x0f) << 8))
                second_value = self._complement_of_two_12_bit(sample_bytes[2] | ((sample_bytes[1] & 0xf0) << 4))

                samples.append([second_value, first_value])
        return samples

    def _sample_range_to_byte_range(self, from_sample, to_sample):
        unrounded_from_byte = int(from_sample * 1.5)
        from_byte = unrounded_from_byte - (unrounded_from_byte % 3)
        unrounded_to_byte = int(to_sample * 1.5)
        to_byte = unrounded_to_byte + (3 - unrounded_to_byte % 3)
        return from_byte, to_byte

    def _complement_of_two_12_bit(self, raw_value):
        sign_bit = (raw_value & 0x800) >> 11
        rest = raw_value & 0x7ff
        return -2048 * sign_bit + rest
