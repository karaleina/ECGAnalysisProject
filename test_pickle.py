from os import path
from matplotlib import pyplot as plt
import numpy as np
import pprint, pickle

from AF.parsers import ecg_recording_parser_af
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser
from AF.simple_medical_analysers import wavelet_analysis


pkl_file = open('database/af_data/04746.pkl', 'rb')
data1 = pickle.load(pkl_file)
pkl_file.close()

print(data1)

plt.close("all")

for i in range(len(data1["channel1"])):

    if (i > -1):
        plt.ion()
        plt.figure(i).clear()
        plt.plot(data1["channel1"][i].get_signal())
        plt.plot(data1["channel0"][i].get_signal())
        plt.pause(0.5)
        plt.close("all")




# data1 = {'a': [1, 2.0, 3, 4+6j],
#          'b': ('string', u'Unicode string'),
#          'c': None}
#
# selfref_list = [1, 2, 3]
# selfref_list.append(selfref_list)
#
# output = open('data.pkl', 'wb')
#
# # Pickle dictionary using protocol 0.
# pickle.dump(data1, output)
#
# # Pickle the list using the highest protocol available.
# pickle.dump(selfref_list, output, -1)
#
# output.close()
#

#
# pkl_file = open('data.pkl', 'rb')
#
# data1 = pickle.load(pkl_file)
# print(data1)
#
# data2 = pickle.load(pkl_file)
# print(data2[3][3][3])
#
# pkl_file.close()


