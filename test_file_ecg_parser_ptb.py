from AF.parsers import ecg_recording_parser_for_ptb, ecg_recording_data_parser
from AF.model import record

from matplotlib import pyplot as plt
from os import path


parser_ptb = ecg_recording_parser_for_ptb.ECGRecordingPTBDataParser()
parser_mit_bih_af = ecg_recording_data_parser.ECGRecordingDataParser()

file_ptb_name_dat = path.join("downloads", "ptb", "patient105", "s0303lre.dat")
file_af_name_dat = path.join("downloads", "af", "04126.dat")

for channel_no in [0, 1]:
    signal_af = parser_mit_bih_af.parse(file_af_name_dat, from_sample=0, to_sample=3000)

    plt.figure(channel_no)
    plt.plot(signal_af)
    plt.title("channel " + str(channel_no))
    plt.show()

for channel_no in [6, 7]:
    signal = parser_ptb.parse(file_ptb_name_dat, channel_no=channel_no, modulo=12, from_sample=0, to_sample=120000)

    print(signal)

    plt.figure(channel_no)
    plt.plot(signal)
    plt.title("channel " + str(channel_no))
    plt.show()
