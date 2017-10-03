#shape(6, N)
from os import path
import pickle

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

