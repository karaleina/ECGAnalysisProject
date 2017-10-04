#shape(6, N)
from os import path
import pickle
from matplotlib import pyplot as plt


def create_af_dataset():
    file_with_all_names = path.join("downloads", "records_names_af")
    file_object = open(file_with_all_names, "r")

    all_dataset =  {"channel0": [],
                    "channel1": []}

    for file in file_object:

        try:
            file = file.replace("\n", "")

            # Opening file
            filepath_pickle = path.join("database", "af_corrected3_data", "new_" + file + ".pkl")
            pkl_file = open(filepath_pickle, 'rb')
            current_dataset = pickle.load(pkl_file)
            pkl_file.close()

            all_dataset["channel1"] += current_dataset["channel1"]
            all_dataset["channel0"] += current_dataset["channel0"]

        except FileNotFoundError:
            print("Nie ma pliku new_" + file + ".pkl")

        print("Len", len(all_dataset["channel0"]))

    return all_dataset

def create_ptb_dataset():
    file_with_all_names = path.join("downloads", "ptb","patients.txt" )
    file_object = open(file_with_all_names, "r")

    all_dataset = {"channel0": [],
                   "channel1": []}

    for file in file_object:

        dir, file = file.split("/")
        try:
            file = file.replace("\n", "")

            # Opening file
            filepath_pickle = path.join("database", "norm_data", dir + "_" + file + ".pkl")
            pkl_file = open(filepath_pickle, 'rb')
            current_dataset = pickle.load(pkl_file)
            pkl_file.close()

            all_dataset["channel1"] += current_dataset["channel1"]
            all_dataset["channel0"] += current_dataset["channel0"]

        except FileNotFoundError:
            print("Nie ma pliku new_" + file + ".pkl")

        print("Len", len(all_dataset["channel0"]))

    return all_dataset

all_af_dataset = create_af_dataset()
all_ptb_dataset = create_ptb_dataset()
all_dataset = {"channel0" : [],
               "channel1" : [],
               "diagnose" : [] # 1 af, 0 norm
}

for i in range(len(all_af_dataset["channel0"])):

    if all_af_dataset["channel0"][i].get_signal() != None and all_af_dataset["channel0"][i].get_signal() != None:
        all_dataset["channel0"].append(all_af_dataset["channel0"][i])
        all_dataset["channel1"].append(all_af_dataset["channel1"][i])
        all_dataset["diagnose"].append(1)

    if all_ptb_dataset["channel0"][i].get_signal() != None and all_ptb_dataset["channel0"][i].get_signal() != None:
        all_dataset["channel0"].append(all_ptb_dataset["channel0"][i])
        all_dataset["channel1"].append(all_ptb_dataset["channel1"][i])
        all_dataset["diagnose"].append(0)

        # plt.ion()
        # plt.figure(2)
        # plt.subplot(2,1,1).cla()
        # plt.plot(all_ptb_dataset["channel0"][i].get_signal())
        # plt.subplot(2,1,2).cla()
        # plt.plot(all_ptb_dataset["channel1"][i].get_signal())
        # plt.pause(0.05)

        #         plt.ion()
        # plt.figure(1)
        # plt.subplot(2,1,1).cla()
        # plt.plot(all_af_dataset["channel0"][i].get_signal())
        # #
        # # plt.subplot(2,1,2).cla()
        # # plt.plot(all_af_dataset["channel1"][i].get_signal())
        # #
        # # plt.figure(2)
        # # plt.subplot(2,1,1).cla()
        # # plt.plot(all_ptb_dataset["channel0"][i*2].get_signal())
        # # plt.subplot(2,1,2).cla()
        # # plt.plot(all_ptb_dataset["channel1"][i * 2].get_signal())
        # #
        # # plt.pause(0.1)

diagnose_list = all_dataset["diagnose"]
print(diagnose_list)
indices_of_af = [i for i, x in enumerate(diagnose_list) if x == 1]
indices_of_ptb = [i for i, x in enumerate(diagnose_list) if x == 0]
print(len(indices_of_af))
print(len(indices_of_ptb))
print(len(diagnose_list))






