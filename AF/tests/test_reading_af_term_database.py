import pickle
from matplotlib import pyplot as plt


pkl_file = open("database/af_term.pkl", 'rb')
dataset = pickle.load(pkl_file)
pkl_file.close()

for patient in dataset:
    patient_dataset = dataset[patient]

    for index in range(len(patient_dataset["channel1"])):

        signal0 = patient_dataset["channel0"][index].get_signal()
        signal1 = patient_dataset["channel1"][index].get_signal()

        plt.ion()
        plt.figure(1).clear()
        plt.subplot(2,1,1)
        plt.plot(signal0)

        plt.subplot(2,1,2)
        plt.plot(signal1)
        plt.pause(0.5)

