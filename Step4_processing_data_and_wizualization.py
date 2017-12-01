from Step2_reading_and_correcting import read_with_pickle
from AF.tools.pca_tools import PCADimensionAnalyser
from matplotlib import pyplot as plt
import numpy as np
import pywt
from PyEMD import EMD
from AF.simple_medical_analysers.wavelet_analysis import  DWTWaveletAnalyser


def transform_dataset_into_pcas_datasets(dataset):
    new_dataset = dataset.copy()
    for patient in dataset:
        list_rr_channel0 = dataset[patient]["channel0"]
        list_rr_channel1 = dataset[patient]["channel1"]
        for index in range(len(list_rr_channel0)):

            # PCA
            input_PCA_dataset = [[el1, el2] for el1, el2 in zip(
                list_rr_channel0[index].get_signal(), list_rr_channel1[index].get_signal())]
            pca_a = PCADimensionAnalyser()
            pca_a.calculate_new_dimension(input_PCA_dataset, number_of_components=2)
            output_PCA_dataset = pca_a.get_new_dimension(input_PCA_dataset)
            signal_pca0 = [el[0] for el in output_PCA_dataset]
            signal_pca1 = [el[1] for el in output_PCA_dataset]


            # Updating
            new_dataset[patient]["channel0"][index].set_signal(signal_pca0)
            new_dataset[patient]["channel1"][index].set_signal(signal_pca1)



    return new_dataset


def transform_dataset_into_coeffs_dataset(dataset, wavelet="db2"):
    new_dataset = dataset
    for patient in dataset:
        list_rr_channel0 = dataset[patient]["channel0"]
        list_rr_channel1 = dataset[patient]["channel1"]
        dataset[patient]["coeffs"] = np.empty((len(list_rr_channel0),2))
        for index in range(len(list_rr_channel0)):
            signal0 = list_rr_channel0[index].get_signal()
            signal1 = list_rr_channel1[index].get_signal()



            # Calculate coeffs
            dwt_a = DWTWaveletAnalyser()
            norm_coeff0 = dwt_a.get_wavelet_af_energy(signal0, frequency=128, wavelet=wavelet)
            norm_coeff1 = dwt_a.get_wavelet_af_energy(signal1, frequency=128, wavelet=wavelet)

            # Updating
            new_dataset[patient]["coeffs"][index, :] = [norm_coeff0, norm_coeff1]



            # TODO Co zrobic z tym, ze wtedy nie bedzie mial pacjent w tym zalamku coeffsow
            pass
    return new_dataset


def calculate_emd_and_show(dataset):
    dictionary = {"aftdb": {"var":[],
                          "std":[]},
                  "ptb":   {"var":[],
                          "std":[]}
                  }
    for patient in dataset:
        list_rr_channel0 = dataset[patient]["channel0"]
        list_rr_channel1 = dataset[patient]["channel1"]

        if dataset[patient]["diagnose"] == "ptb":
            for index in range(len(list_rr_channel0)):
                signal0 = list_rr_channel0[index].get_signal()
                signal1 = list_rr_channel1[index].get_signal()

                emd = EMD()
                eIMFs0 = emd.emd(signal0)
                number_of_imfs = len(eIMFs0)
                for i in range(number_of_imfs):
                    fig = plt.figure(1)
                    plt.subplot(number_of_imfs,1,i+1)
                    plt.plot(eIMFs0[i, :])
                    fig.savefig(patient + "_" + str(dataset[patient]["diagnose"]) +  "_chann_0_" + ".jpg")

                emd = EMD()
                eIMFs1 = emd.emd(signal1)
                number_of_imfs = len(eIMFs1)
                for i in range(number_of_imfs):
                    fig = plt.figure(2)
                    plt.subplot(number_of_imfs, 1, i + 1)
                    plt.plot(eIMFs1[i, :])
                    fig.savefig(patient + "_" + str(dataset[patient]["diagnose"]) + "_chann_1_" + ".jpg")

                break




if __name__ == "__main__":

    directory = "database/step3"
    X_test = read_with_pickle(directory + "/" + "X_test.pkl")
    X_train = read_with_pickle(directory + "/" + "X_train.pkl")

    #calculate_emd_and_show(X_test)

    # PCA pca przynosi odwrotny skutek!!!!!!!!!!! REZYGNUJĘ
    # X_test_pcas = transform_dataset_into_pcas_datasets(X_test)
    # X_train_pcas = transform_dataset_into_pcas_datasets(X_train)

    # Wavelets
    # TODO Number of samples!!!
    #dbs = ["db" + str(i) for i in range(1, 21)]
    #syms = ["sym" + str(i) for i in range(2, 21)]
    #coifs = ["coif" + str(i) for i in range(1,6)]

    #for index, wavelet in enumerate(pywt.wavelist(family=None, kind='all')):
    wavelet = "db2"
    X_test_wavelets_coeffs = transform_dataset_into_coeffs_dataset(X_test, wavelet=wavelet)
    X_train_wavelets_coeffs = transform_dataset_into_coeffs_dataset(X_train, wavelet=wavelet)

    #Data wizualization
    plt.figure(2)
    for patient_name in X_test_wavelets_coeffs:
        coeffs = X_test_wavelets_coeffs[patient_name]["coeffs"]
        color = "red" if X_test_wavelets_coeffs[patient_name]["diagnose"] == "aftdb" else "blue"
        plt.scatter(x=coeffs[:,0], y=coeffs[:,1], color=color)
        plt.title("Falka " + wavelet)

    # plt.figure(2)
    # for patient_name in X_train_wavelets_coeffs:
    #     coeffs = X_train_wavelets_coeffs[patient_name]["coeffs"]
    #     color = "red" if X_test_wavelets_coeffs[patient_name]["diagnose"] == "aftdb" else "blue"
    #     plt.scatter(x=coeffs[:,0], y=coeffs[:,1], color=color)
    #     plt.title("Falka" + wavelet)
    plt.show()

    # TODO testy NN z obecnym stanem prac
    # TODO falki
    # TODO emd
