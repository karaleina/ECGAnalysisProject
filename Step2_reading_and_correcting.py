import pickle
from matplotlib import pyplot as plt

def read_with_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


aftdb = read_with_pickle("database/step1/aftdb.pkl")
# n1: OK
# n2: OK
# n3: CALE ZLE????


for patient_record in aftdb:

    if not patient_record in ['n01', 'n02']:
        temp_patient_dataset = aftdb[patient_record]

        new_patient_dataset = {"channel0": [],
                               "channel1": []}

        for index in range(len(temp_patient_dataset["channel0"])):

            # SHOWING
            plt.ion()
            plt.figure(1).clear()
            plt.subplot(2,1,1)
            plt.plot(aftdb[patient_record]["channel0"][index].get_signal())
            plt.subplot(2,1,2)
            plt.plot(aftdb[patient_record]["channel1"][index].get_signal())
            plt.suptitle(patient_record + "_" + str(index))

            # TODO correcting
            user_input = input("Type 'del' if intend to delete this RR segment")
            if user_input=="del":
                del temp_patient_dataset["channel0"][index]
                del temp_patient_dataset["channel1"][index]
                print("Dlugosc listy zalamkow dla pacjenta " + patient_record + " : " + str(len(temp_patient_dataset["channel0"])))

            plt.pause(0.05)
