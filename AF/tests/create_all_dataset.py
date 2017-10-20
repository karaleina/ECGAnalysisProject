from os import path
import pickle
from AF.simple_medical_analysers import wavelet_analysis
import numpy as np
from sklearn import datasets, linear_model
from neural_model_functions.simple_neural_models import plot_decision_boundary, build_model, predict
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

def create_all_dataset():

    all_af_dataset = create_af_dataset()
    all_ptb_dataset = create_ptb_dataset()
    all_dataset = {"channel0": [],
                   "channel1": [],
                   "diagnose": []  # 1 af, 0 norm
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

    return all_dataset

def create_wavelet_dataset(dataset_with_diagnose, number_of_samples, wavelet):

    X_dataset = {"coeffs": [],
                 "diagnose": []
                 }

    for i in range(len(dataset_with_diagnose["channel0"])):
        # Unpacking signal
        chann0 = dataset_with_diagnose["channel0"][i].get_signal()
        chann1 = dataset_with_diagnose["channel1"][i].get_signal()

        signal_length = len(chann0)


        if signal_length > number_of_samples:

            # Cropping the center of the signal
            center = signal_length/2 + 1
            new_chann0 = chann0[int(center - number_of_samples / 2):int(center + number_of_samples / 2)]
            new_chann1 = chann1[int(center - number_of_samples / 2):int(center + number_of_samples / 2)]

            # Filtration
            signal_chann_0_filtered = wavelet_analysis.filter_signal(new_chann0, wavelet="db6")
            signal_chann_1_filtered = wavelet_analysis.filter_signal(new_chann1, wavelet="db6")

            # Wavelet coeffs
            (norm_coeffs0, reconstructed_chann0) = wavelet_analysis.get_AF_coeffs_and_AF_signal(signal_chann_0_filtered, wavelet=wavelet)
            (norm_coeffs1, reconstructed_chann0) = wavelet_analysis.get_AF_coeffs_and_AF_signal(signal_chann_1_filtered, wavelet=wavelet)

            # print(norm_coeffs0[0])
            # print(norm_coeffs1[0])

            # Adding to big dataset
            X_dataset["coeffs"].append(norm_coeffs0[0] + norm_coeffs1[0])
            X_dataset["diagnose"].append(dataset_with_diagnose["diagnose"][i])


        else:
            print("Zbyt krótki sygnał")

    return X_dataset


# CREATING DATASET
dataset_with_diagnose = create_all_dataset()

# CREATING WAVELET DATASET
wavelet_DATA = create_wavelet_dataset(dataset_with_diagnose, number_of_samples=90, wavelet="db2")

# NEURAL NETWORK DATASET

X = np.array(wavelet_DATA["coeffs"])
y = np.array(wavelet_DATA["diagnose"])

# Linear regression
clf = linear_model.LogisticRegressionCV()

index = 0
for i in range(6):
    for j in range(6):
        if i == j:
            break
        else:

            plt.ion()

            X_for_regression = X[:, [j, i]]
            plt.figure(0).clear()
            plt.title(str(j) + ", " + str(i))
            plt.scatter(X_for_regression[:, 0], X_for_regression[:, 1], s=40, c=y, cmap="cool")#plt.cm.Spectral)
            index += 1

            plt.pause(2)


# Building a neural network
hdim = 9

print(X)
model = build_model(nn_input_dim=6, nn_hdim=hdim, nn_output_dim=2,
                    X=X, y=y, num_examples=len(X),
                    reg_lambda=0.01, epsilon=0.01,
                    num_passes=20000)

print("Otrzymany model SNN: ", model)

predictions = predict(model, X)
difference = np.array(predictions)-np.array(y)

bad_classified = [i for i, x in enumerate(difference) if x != 0]

print(len(bad_classified))
print(len(predictions))
print(len(bad_classified)/len(predictions))

# TESTS

# print(X)
# print(len(X["diagnose"]))
# diagnose_list = all_dataset["diagnose"]
# print(diagnose_list)
#
# indices_of_af = [i for i, x in enumerate(diagnose_list) if x == 1]
# indices_of_ptb = [i for i, x in enumerate(diagnose_list) if x == 0]
#
# print(len(indices_of_af))
# print(len(indices_of_ptb))
# print(len(diagnose_list))






