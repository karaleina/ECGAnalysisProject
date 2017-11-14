from PyEMD import EMD
import numpy as np
from matplotlib import pyplot as plt
from Step2_reading_and_correcting import read_with_pickle


# Test signal
aftdb = read_with_pickle("/home/karolina/PycharmProjects/atrialFibrillationAnalysisProject/database/step2/aftdb_corrected.pkl")
ptb = read_with_pickle("/home/karolina/PycharmProjects/atrialFibrillationAnalysisProject/database/step1/ptb.pkl")
#
#
# emd = EMD()
#
# for patient_record in aftdb:
#     temp_patient_dataset = aftdb[patient_record]
#     s = None
#     for index in range(len(temp_patient_dataset["channel0"])):
#         s = aftdb[patient_record]["channel1"][index].get_signal()
#         plt.figure(0)
#         plt.plot(s)
#         # EMD test
#         eIMFs = emd.emd(s)
#         print(eIMFs.shape)
#         number_of_imfs = len(eIMFs)
#         for i in range(number_of_imfs):
#             plt.figure(1)
#             plt.subplot(number_of_imfs,1,i+1)
#             plt.plot(eIMFs[i, :])
#
#         plt.show()