from AF.parsers import ecg_recording_parser_for_ptb
from matplotlib import pyplot as plt

parser = ecg_recording_parser_for_ptb.ECGRecordingPTBDataParser()
file_name_dat = "s0303lre.dat"

for channel_no in range(0, 11):
    signal = parser.parse(file_name_dat, channel_no=channel_no, modulo=12, from_sample=0, to_sample=120000)
    print(signal)

    plt.figure()
    plt.plot(signal)
    plt.title("channel " + str(channel_no))
    plt.show()
