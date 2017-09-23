from os import path
from matplotlib import pyplot as plt
from AF.parsers import ecg_recording_parser_af

import wfdb
import numpy as np

file_path = path.join("downloads", "af", "04048")
record = wfdb.rdsamp(file_path, sampto = 556777+3000 , sampfrom=556777-3000)

#a= wfdb.rdann('04746','atr',pbdir='afdb', sampto=103000)
annotation = wfdb.rdann(file_path, 'atr')
annsamp = annotation.__dict__["annsamp"]
names = annotation.__dict__["aux"]
print(annsamp, names)


# SIGNALS
parser = ecg_recording_parser_af.ECGRecordingDataParser()
signals_af = np.array(parser.parse(dat_file_name=file_path+'.dat', from_sample=556777-3000, to_sample=556777+3000))

signal_0 = signals_af[:, 0]
signal_1 = signals_af[:, 1]

plt.figure(4)
plt.subplot(2,1,1)
plt.plot(signal_0)
plt.subplot(2,1,2)
plt.plot(signal_1)


#
# signal_0 = wavelet_analysis.filter_signal(signal_0, wavelet="db6", highcut=True)
# signal_1 = wavelet_analysis.filter_signal(signal_1, wavelet="db6", highcut=True)


#wfdb.plotrec(record, annotation=annotation, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

plt.show()
#

#
# afib_samples = []
# for index, samp in enumerate(annsamp):
#     if names[index] == "(AFIB":
#         afib_samples.append(samp)
#
# print(afib_samples)
#
#
