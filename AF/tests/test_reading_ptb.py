from AF.parsers import ecg_recording_parser_ptb
from os import path
from matplotlib import pyplot as plt

file_path = path.join("downloads", "ptb", "patient105","s0303lre.dat")

ptb_parser = ecg_recording_parser_ptb.ECGRecordingPTBDataParser()
signal = ptb_parser.parse(dat_file_name=file_path, channel_no=6, modulo=12, from_sample=0, to_sample=100000, invert=False)

plt.plot(signal)
plt.show()