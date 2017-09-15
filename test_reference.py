from os import path

import numpy as np
from AF.algorithms import r_wave_detection
from matplotlib import pyplot as plt

from AF.parsers import ecg_recording_data_parser

record_normal = path.join("downloads", "nsrdb", "16420")

parser = ecg_recording_data_parser.ECGRecordingDataParser()
signals = np.array(parser.parse(str(record_normal) + ".dat", 0, 1000))

signals[:, 0] = signals[:, 0].ravel()
signals[:, 1] = signals[:, 1].ravel()

qrs_detect = r_wave_detection.Christov()

R_waves_0 = qrs_detect.detect_r_waves(signals[:, 0])
R_waves_1 = qrs_detect.detect_r_waves(signals[:, 1])

print(R_waves_0)

plt.plot(signals[:,0])
plt.plot(R_waves_0, np.ones(len(R_waves_0)), "*")

plt.show()

