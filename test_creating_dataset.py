from os import path
import pickle
from AF.analyzers.dimensionAnalyzer import PCADimensionAnalyser
from AF.simple_medical_analysers import wavelet_analysis
import numpy as np
from sklearn import datasets, linear_model
from neural_model_functions.simple_neural_models import plot_decision_boundary, build_model, predict
from matplotlib import pyplot as plt
from PyEMD import EMD
import numpy as np
from matplotlib import pyplot as plt


def create_af_dataset(file_with_all_names, pickle_dir):

    file_object = open(file_with_all_names, "r")
    all_dataset =  {"channel0": [],
                    "channel1": []}

    for file in file_object:

        try:
            file = file.replace("\n", "")

            # Opening file
            filepath_pickle = path.join(pickle_dir, "new_" + file + ".pkl")
            pkl_file = open(filepath_pickle, 'rb')
            current_dataset = pickle.load(pkl_file)
            pkl_file.close()

            all_dataset["channel1"] += current_dataset["channel1"]
            all_dataset["channel0"] += current_dataset["channel0"]

        except FileNotFoundError:
            print("Nie ma pliku new_" + file + ".pkl")

        print("Len", len(all_dataset["channel0"]))

    return all_dataset


def create_ptb_dataset(file_with_all_names, pickle_dir):

    file_object = open(file_with_all_names, "r")

    all_dataset = {"channel0": [],
                   "channel1": []}

    for file in file_object:

        dir, file = file.split("/")
        try:
            file = file.replace("\n", "")

            # Opening file
            filepath_pickle = path.join( pickle_dir, dir + "_" + file + ".pkl")
            pkl_file = open(filepath_pickle, 'rb')
            current_dataset = pickle.load(pkl_file)
            pkl_file.close()

            all_dataset["channel1"] += current_dataset["channel1"]
            all_dataset["channel0"] += current_dataset["channel0"]

        except FileNotFoundError:
            print("Nie ma pliku new_" + file + ".pkl")

        print("Len", len(all_dataset["channel0"]))

    return all_dataset


def create_wavelet_dataset(dataset_with_diagnose, wavelet, cropping=False, number_of_samples=None, make_pca=True):

    X_dataset = {"coeffs": [],
                 "diagnose": []}

    for i in range(len(dataset_with_diagnose["channel0"])):
        # Unpacking signal
        chann0 = dataset_with_diagnose["channel0"][i].get_signal()
        chann1 = dataset_with_diagnose["channel1"][i].get_signal()
        signal_length = len(chann0)

        if signal_length > number_of_samples:
            if cropping == True:

                    # Cropping the center of the signal
                    center = signal_length/2 + 1
                    chann0 = chann0[int(center - number_of_samples / 2):int(center + number_of_samples / 2)]
                    chann1 = chann1[int(center - number_of_samples / 2):int(center + number_of_samples / 2)]

            # Filtration
            signal_chann_0_filtered = wavelet_analysis.filter_signal(chann0, wavelet="db6")
            signal_chann_1_filtered = wavelet_analysis.filter_signal(chann1, wavelet="db6")

            # PCA
            if make_pca==True:
                data_matrix = np.empty((len(signal_chann_1_filtered), 2))
                data_matrix[:,0] = signal_chann_0_filtered
                data_matrix[:,1] = signal_chann_1_filtered

                #print(data_matrix)
                a_pca = PCADimensionAnalyser()
                a_pca.calculate_new_dimension(train_data_matrix=data_matrix, pca_components=2)
                new_data_matrix = a_pca.get_new_dimension(test_data_matrix=data_matrix)

                plt.ion()
                plt.figure(-1).clear()
                plt.subplot(2,2,1)
                plt.plot(signal_chann_0_filtered)
                plt.title("Channel0")
                plt.subplot(2,2,2)
                plt.plot(signal_chann_1_filtered)
                plt.title("Channel1")
                plt.subplot(2,2,3)
                plt.plot(new_data_matrix[:,0])
                plt.title("PCA1")
                plt.subplot(2,2,4)
                plt.plot(new_data_matrix[:,1])
                plt.title("PCA2")

                # plt.figure(-2).clear()
                # emd = EMD()
                # IMFs = emd.emd(new_data_matrix[:,1])
                # plt.plot(new_data_matrix[:,1])
                #
                # for index, imf in enumerate(IMFs):
                #     plt.figure(index)
                #     plt.plot(imf)
                #     plt.title("IMF nr" + str(index) + "for channel0")
                #
                plt.pause(1)


                signal_chann_0_filtered = new_data_matrix[:,0]
                signal_chann_1_filtered = new_data_matrix[:,1]


            # Wavelet coeffs
            coeff_energy0 = wavelet_analysis.get_AF_energy(signal_chann_0_filtered, wavelet=wavelet)
            coeff_energy1 = wavelet_analysis.get_AF_energy(signal_chann_1_filtered, wavelet=wavelet)

            # Adding to big dataset
            X_dataset["coeffs"].append([coeff_energy0, coeff_energy1])
            X_dataset["diagnose"].append(dataset_with_diagnose["diagnose"][i])

        else:
            print("Zbyt krótki sygnał, aby przyciąć")

    return X_dataset

def create_test_and_train_dataset(wavelet_dataset):

    test_dataset  = {"coeffs": [],
                     "diagnose": []}
    train_dataset = {"coeffs": [],
                     "diagnose": []}

    for index in range(len(wavelet_dataset["coeffs"])):
        if index % 3 == 0:
            test_dataset["coeffs"].append(wavelet_dataset["coeffs"][index])
            test_dataset["diagnose"].append(wavelet_dataset["diagnose"][index])
        else:
            train_dataset["coeffs"].append(wavelet_dataset["coeffs"][index])
            train_dataset["diagnose"].append(wavelet_dataset["diagnose"][index])

    return train_dataset, test_dataset

# Creating dataset
file_with_all_names = path.join("downloads", "records_names_af")
pickle_dir = path.join("database", "af_corrected3_data")
all_af_dataset = create_af_dataset(file_with_all_names=file_with_all_names, pickle_dir=pickle_dir)

file_with_all_names = path.join("downloads", "ptb","patients.txt" )
pickle_dir = path.join("database", "norm_data")
all_ptb_dataset = create_ptb_dataset(file_with_all_names=file_with_all_names, pickle_dir=pickle_dir)

# Making big dataset
all_dataset = {"channel0": [], "channel1": [], "diagnose": []}

for i in range(len(all_af_dataset["channel0"])):
    if all_af_dataset["channel0"][i].get_signal() != None and all_af_dataset["channel0"][i].get_signal() != None:
        all_dataset["channel0"].append(all_af_dataset["channel0"][i])
        all_dataset["channel1"].append(all_af_dataset["channel1"][i])
        all_dataset["diagnose"].append(1)

    if all_ptb_dataset["channel0"][i].get_signal() != None and all_ptb_dataset["channel0"][i].get_signal() != None:
        all_dataset["channel0"].append(all_ptb_dataset["channel0"][i])
        all_dataset["channel1"].append(all_ptb_dataset["channel1"][i])
        all_dataset["diagnose"].append(0)

wavelet_DATA = create_wavelet_dataset(dataset_with_diagnose=all_dataset, wavelet="sym2", cropping=False, number_of_samples=120)

# CREATING TEST AND TRAIN DATASET:
train_DATA, test_DATA = create_test_and_train_dataset(wavelet_DATA)

# NEURAL NETWORK DATASET

X_train = np.array(train_DATA["coeffs"])
y_train = np.array(train_DATA["diagnose"])

X_test = np.array(test_DATA["coeffs"])
y_test = np.array(test_DATA["diagnose"])


# Linear regression
clf = linear_model.LogisticRegressionCV()

index = 0
for i in range(2):
    for j in range(2):
        if i == j:
            break
        else:

            X_for_regression = X_train[:, [j, i]]
            plt.figure(0).clear()
            plt.title(str(j) + ", " + str(i))
            plt.scatter(X_for_regression[:, 0], X_for_regression[:, 1], s=40, c=y_train, cmap="cool")#plt.cm.Spectral)
            plt.colorbar()
            index += 1

            plt.show()


# Building a neural network
hdim = 5

print(X_train)
model = build_model(nn_input_dim=2, nn_hdim=hdim, nn_output_dim=2,
                    X=X_train, y=y_train, num_examples=len(X_train),
                    reg_lambda=0.001, epsilon=0.001,
                    num_passes=20000)

print("Otrzymany model SNN: ", model)

predictions = predict(model, X_test)
difference = np.array(predictions)-np.array(y_test)

bad_classified = [i for i, x in enumerate(difference) if x != 0]

print(len(bad_classified))
print(len(predictions))
print(len(bad_classified)/len(predictions))

plot_decision_boundary(lambda x: predict(model, x), X_test, y_test)
plt.title("Decision Boundary for hidden layer size " + str(hdim))
plt.show()


