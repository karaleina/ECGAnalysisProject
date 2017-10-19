from AF.parsers import ecg_recording_parser_af
from matplotlib import pyplot as plt

from AF.simple_medical_analysers import wavelet_analysis
import numpy as np
from os import path
import wfdb


from os import path
from matplotlib import pyplot as plt
from AF.parsers import ecg_recording_parser_af

import wfdb
import numpy as np

dir = path.join("downloads", "af_term")
i = 0
with open("records_names") as patients_list:

    for patient_id in patients_list:
        patient_id = patient_id.replace("\n", "")
        subdir, record_no = patient_id.split("/")

        if subdir != "learning-set":
            pass

        else:

            filepath = path.join(dir, subdir, record_no)
            record = wfdb.rdsamp(filepath)

            print("Dlugosc sygnalu :", record.siglen)

            # Wczytuje ostatnie 2000 probek sygnalu
            signals, fields = wfdb.srdsamp(filepath, sampfrom= record.siglen - 2001, sampto= record.siglen - 1, channels=None, pbdir=None)


            plt.figure(i).clear()
            plt.plot(signals[:,1])
            plt.title(patient_id)
            i += 1

    plt.show()






# #04048  2154473 nie ma afibu a jest afib
#
# file_path = path.join("downloads", "af", "04048")
# record = wfdb.rdsamp(file_path, sampto = 556777+3000 , sampfrom=556777-3000)
#
# #a= wfdb.rdann('04746','atr',pbdir='afdb', sampto=103000)
# annotation = wfdb.rdann(file_path, 'atr')
# annsamp = annotation.__dict__["annsamp"]
# names = annotation.__dict__["aux"]
# print(annsamp, names)
#
# afib_samples = []
# for index, samp in enumerate(annsamp):
#     if names[index] == "(AFIB":
#         afib_samples.append(samp)
#
#
# interval_in_samples = 1000
#
# for stop in afib_samples:
#
#
#     while True:
#
#         start = stop - interval_in_samples
#
#         # SIGNALS
#         parser = ecg_recording_parser_af.ECGRecordingDataParser()
#         signals_af = np.array(parser.parse(dat_file_name=file_path+".dat", from_sample=start, to_sample=stop))
#
#         signal_0 = signals_af[:, 0]
#         signal_1 = signals_af[:, 1]
#
#         signal_0 = wavelet_analysis.filter_signal(signal_0, wavelet="db6", highcut=True)
#         signal_1 = wavelet_analysis.filter_signal(signal_1, wavelet="db6", highcut=True)
#
#         plt.ion()
#         plt.figure(0).clear()
#         plt.subplot(2,1,1)
#         plt.plot(signal_0)
#         plt.subplot(2,1,2)
#         plt.plot(signal_1)
#         plt.pause(0.005)
#
#         print("start_sample = ", stop, "end_sample = ", start)
#
#         print(annsamp, names)
#         command = input("Press Enter to continue...")
#
#         if command == "next":
#             break
#
#         stop += interval_in_samples
