from AF.parsers import ecg_recording_parser_ptb, ecg_recording_parser_af
from AF.model import record

from matplotlib import pyplot as plt
import numpy as np
from os import path


parser_ptb = ecg_recording_parser_ptb.ECGRecordingPTBDataParser()
parser_mit_bih_af = ecg_recording_parser_af.ECGRecordingDataParser()

file_ptb_name_dat = path.join("downloads", "ptb", "patient104", "s0306lre.dat")
file_af_name_dat = path.join("downloads", "af", "04048.dat")



for channel_no in [0]:

    signal_af = np.array(parser_mit_bih_af.parse(file_af_name_dat, from_sample=0, to_sample=3000))

    plt.figure(0)
    plt.plot(signal_af[:,0].ravel())


    plt.figure(1)
    plt.plot(signal_af[:, 1].ravel())


for channel_no in [6, 7]:
    signal = parser_ptb.parse(file_ptb_name_dat, channel_no=channel_no, invert=False, modulo=12, from_sample=0, to_sample=120000)

    print(signal)

    plt.figure(channel_no)
    plt.plot(signal)
    plt.title("channel " + str(channel_no))

plt.show()
