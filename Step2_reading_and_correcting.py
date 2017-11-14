import pickle
from matplotlib import pyplot as plt


def read_with_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_with_pickle(data, pickle_file):
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    aftdb = read_with_pickle("database/step1/aftdb_corrected.pkl")
    # n3: CALE ZLE????
    # s9: fala P?

    list_names = ['n0' + str(i) if i < 10 else 'n' + str(i) for i in range(1, 11)]
    list_names += ['s0' + str(i) if i < 10 else 's' + str(i) for i in range(1, 11)]
    list_names += ['t0' + str(i) if i < 10 else 't' + str(i) for i in range(1, 10)]
    print(list_names)

    for patient_record in aftdb:
        if not patient_record in list_names:
            temp_patient_dataset = aftdb[patient_record]
            for index in range(len(temp_patient_dataset["channel0"])):
                # SHOWING
                plt.ion()
                plt.figure(1).clear()
                plt.subplot(2,1,1)
                plt.plot(aftdb[patient_record]["channel0"][index].get_signal())
                plt.subplot(2,1,2)
                plt.plot(aftdb[patient_record]["channel1"][index].get_signal())
                plt.suptitle(patient_record + "_" + str(index))

                # CORRECTING
                user_input = input("Type 'del' if intend to delete this RR segment")
                if user_input == "del":
                    del temp_patient_dataset["channel0"][index]
                    del temp_patient_dataset["channel1"][index]
                    print("Dlugosc listy zalamkow dla pacjenta " + patient_record + " : " + str(len(temp_patient_dataset["channel0"])))
                    user_input2 = input("Type 'save' if intend to save this corrections")
                    if user_input2 == "save":
                        aftdb[patient_record] = temp_patient_dataset
                        new_file_name_pkl = 'database/step1/aftdb_corrected.pkl'
                        save_with_pickle(aftdb, new_file_name_pkl)
                        print("Zapisano do " + new_file_name_pkl)

                plt.pause(0.05)
