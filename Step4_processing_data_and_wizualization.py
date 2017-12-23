from Step2_reading_and_correcting import read_with_pickle
from AF.tools.pca_tools import PCADimensionAnalyser
from AF.simple_medical_analysers.wavelet_analysis import DWTWaveletAnalyser
from AF.analyzers.qualityEvaluation import calculate_quality_of_classification
from neural_model_functions.simple_neural_models import plot_decision_boundary, build_model, predict
from neural_model_functions import SNN
from matplotlib import pyplot as plt
import numpy as np
from PyEMD import EMD
from sklearn.svm import LinearSVC
from AF import knn_algorithm, fft_module


# TODO 
def calculate_ROC_params(predicted_y_test_without_class, y_real, min_threshold, max_threshold, number_of_samples, plotting=True):
    thresholds = np.linspace(start=min_threshold, stop=max_threshold, num=number_of_samples)
    sensitivities = np.empty_like(thresholds)
    specifities = np.empty_like(thresholds)

    for index_thresh, threshold in enumerate(thresholds):
        predicted_y_test_classified = [0 if element < threshold else 1 for element in predicted_y_test_without_class]
        quality = calculate_quality_of_classification(y_real=y_real, y_predictions=predicted_y_test_classified)
        specifities[index_thresh] = quality["specifity"]
        sensitivities[index_thresh] = quality["sensitivity"]

    if plotting is True:
        plt.plot(1-specifities, sensitivities, "-o")
        plt.axis([0, 1, 0, 1])
        plt.ylabel("sensitivity")
        plt.xlabel("1-specifity")
        plt.grid()
        plt.show()


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


def transform_dataset_into_coeffs_dataset(dataset, list_of_wavelets):
    X_coeffs_dataset = []
    y_info_dataset = []

    for patient in dataset:
        list_rr_channel0 = dataset[patient]["channel0"]
        list_rr_channel1 = dataset[patient]["channel1"]
        diagnose = dataset[patient]["diagnose"]
        dataset[patient]["coeffs"] = np.empty((len(list_rr_channel0),2))
        for index in range(len(list_rr_channel0)):
            signal0 = list_rr_channel0[index].get_signal()
            signal1 = list_rr_channel1[index].get_signal()
            record_data = []
            for wavelet in list_of_wavelets:
                # Calculate coeffs
                dwt_a = DWTWaveletAnalyser()
                norm_coeff0 = dwt_a.get_wavelet_af_energy(signal0, frequency=128, wavelet=wavelet)
                norm_coeff1 = dwt_a.get_wavelet_af_energy(signal1, frequency=128, wavelet=wavelet)
                record_data.extend((norm_coeff0, norm_coeff1))
            # Updating
            y_info_dataset.append([diagnose, patient])
            X_coeffs_dataset.append(record_data)
    return np.array(X_coeffs_dataset), np.array(y_info_dataset)


def transform_dataset_into_fft_power_dataset(dataset, T_sampling, f_min, f_max):
    X_fft_dataset = []
    y_info_dataset = []

    for patient in dataset:
        list_rr_channel0 = dataset[patient]["channel0"]
        list_rr_channel1 = dataset[patient]["channel1"]
        diagnose = dataset[patient]["diagnose"]
        dataset[patient]["coeffs"] = np.empty((len(list_rr_channel0),2))

        for index in range(len(list_rr_channel0)):
            signal0 = list_rr_channel0[index].get_signal()
            signal1 = list_rr_channel1[index].get_signal()
            record_data = []

            # FFT
            coeff_fft0_normed = fft_module.FFT_analyser.get_fft_coeff(signal0, T_sampling=T_sampling, f_min=f_min, f_max=f_max)
            coeff_fft1_normed = fft_module.FFT_analyser.get_fft_coeff(signal1, T_sampling=T_sampling, f_min=f_min, f_max=f_max)
            record_data.extend((coeff_fft0_normed, coeff_fft1_normed))

            # Updating
            y_info_dataset.append([diagnose, patient])
            X_fft_dataset.append(record_data)
    return np.array(X_fft_dataset), np.array(y_info_dataset)


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


def main(dataset_type="fft", snn_go=False, svm_go=False, knn_go=False,
         number_of_hidden_neurons=None, list_of_wavelets=None):
    # TODO Douczanie samych przypadków trudnych:
    #  - z niezgodną przynależnością -  sieć neuronowa
    # - wytypowanie falek do ogolnej klasyfikacji i douczenia przypadkow trudnych

    # TODO Saving SNN, SVM, (k-means?) results weights
    # TODO EMD dataset

    # -------------------------PARAMS------------------------------------
      # ""wavelets"# "wavelets" # "fft"



    # ---------------- Wizualization parameters ----------------------
    data_visualisation = True
    class_no_1 = 1
    class_no_2 = 2

    # -----------------------CREATING DATASET---------------------------------------
    directory = "database/step3"
    X_test = read_with_pickle(directory + "/" + "X_test.pkl")
    X_train = read_with_pickle(directory + "/" + "X_train.pkl")

    # ---------------------------WAVELET-DATASET----------------------------
    if dataset_type == "wavelets":
        X_test_wavelets_coeffs, y_test_info = transform_dataset_into_coeffs_dataset(X_test, list_of_wavelets)  # , "dmey", "haar")
        X_train_wavelets_coeffs, y_train_info = transform_dataset_into_coeffs_dataset(X_train, list_of_wavelets)  # , "dmey", "haar")

        # Datasets
        X_train_SNN = X_train_wavelets_coeffs
        X_train_SNN = X_train_SNN.astype('float')
        y_train_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_train_info[:, 0]]
        X_test_SNN = X_test_wavelets_coeffs
        X_test_SNN = X_test_SNN.astype('float')
        y_test_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_test_info[:, 0]]

    # --------------------------FFT-DATASET---------------------------------
    if dataset_type == "fft":
        X_test_fft_coeffs, y_test_info = transform_dataset_into_fft_power_dataset(X_test, T_sampling=1 / 128, f_min=6,
                                                                                  f_max=10)
        X_train_fft_coeffs, y_train_info = transform_dataset_into_fft_power_dataset(X_train, T_sampling=1 / 128,
                                                                                    f_min=6, f_max=10)

        # Datasets
        X_train_SNN = X_train_fft_coeffs
        X_train_SNN = X_train_SNN.astype('float')
        y_train_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_train_info[:, 0]]
        X_test_SNN = X_test_fft_coeffs
        X_test_SNN = X_test_SNN.astype('float')
        y_test_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_test_info[:, 0]]

    # -------------------------Classification-----------------------------
    if svm_go is True:
        # ---------------------------- SVM -------------------------------------
        # Training
        # clf = LinearSVC(random_state=0)
        #
        dual_problem = False  # dual=False when n_samples > n_features.
        clf = LinearSVC(C=1.0, class_weight=None, dual=dual_problem, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=100000,
                        multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
                        verbose=0)
        clf.fit(X_train_SNN, y_train_SNN)

        # Testing
        predicted_y_test = clf.decision_function(X_test_SNN)

        # ROC
        calculate_ROC_params(predicted_y_test_without_class=predicted_y_test, y_real=y_test_SNN, min_threshold=-2,
                             max_threshold=2, number_of_samples=2000)
        # predicted_y_test = [0 if element < 0 else 1 for element in predicted_y_test]
        # print(predicted_y_test)
        # Quality
        # quality = calculate_quality_of_classification(y_real=y_test_SNN, y_predictions=predicted_y_test)
        # print("Specyficznosc:", quality["specifity"])
        # print("Czułość:", quality["sensitivity"])

        if data_visualisation is True:
            # -------------- Data visualisation ---------------------------------
            for element, y_label in zip(X_test_SNN, y_test_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Test dataset")

            plt.figure(2)
            for element, y_label in zip(X_train_SNN, y_train_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Train dataset")

            plt.figure(3)
            for index, class_y in enumerate(predicted_y_test):
                color = "blue" if class_y <= 0 else "red"
                plt.scatter(X_test_SNN[index, class_no_1 - 1], X_test_SNN[index, class_no_2 - 1], color=color)
            plt.title("Results")

            plt.show()

    if knn_go is True:
        # --------------------------K-NN Klasyfikacja------------------------
        predicted_y_test = []
        predicted_y_values = []
        for x_test_instance in X_test_SNN:
            neighbours = knn_algorithm.getNeighbours(X_train_SNN, y_train_SNN, x_test_instance, k=5)
            predicted_y_value = knn_algorithm.getPrediction(neighbours, weighted_prediction=True)
            predicted_y_values.append(predicted_y_value)
            predicted_y_test.append(1) if predicted_y_value > 0.5 else predicted_y_test.append(0)

        calculate_ROC_params(predicted_y_test_without_class=predicted_y_values, y_real=y_test_SNN, min_threshold=-2,
                             max_threshold=2, number_of_samples=50)

        quality = calculate_quality_of_classification(y_real=y_test_SNN, y_predictions=predicted_y_test)
        print("Specyficznosc:", quality["specifity"])
        print("Czułość:", quality["sensitivity"])

        if data_visualisation is True:
            # -------------- Data visualisation ---------------------------------
            for element, y_label in zip(X_test_SNN, y_test_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Test dataset")

            plt.figure(2)
            for element, y_label in zip(X_train_SNN, y_train_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Train dataset")

            plt.figure(3)
            for index, class_y in enumerate(predicted_y_test):
                color = "blue" if class_y <= 0 else "red"
                plt.scatter(X_test_SNN[index, class_no_1 - 1], X_test_SNN[index, class_no_2 - 1], color=color)
            plt.title("Results")

            plt.show()

    if snn_go is True:
        # ----------------------------SNN--------------------------------------
        number_of_input_neurons = len(X_train_SNN[0])
        nn = SNN.NeuralNetwork([number_of_input_neurons, number_of_hidden_neurons, 1])
        nn.fit(X_train_SNN, y_train_SNN)
        predicted_y_test = [nn.predict(e) for e in X_test_SNN]

        # ROC
        calculate_ROC_params(predicted_y_test_without_class=predicted_y_test, y_real=y_test_SNN, min_threshold=-0.9,
                             max_threshold=1, number_of_samples=2000)

        # predicted_y_test = [1 if nn.predict(e) > 0.5 else 0 for e in X_test_SNN]
        # quality = calculate_quality_of_classification(y_real=y_test_SNN, y_predictions=predicted_y_test)
        # print("Specyficznosc:", quality["specifity"])
        # print("Czułość:", quality["sensitivity"])

        if data_visualisation is True:
            # -------------- Data visualisation ---------------------------------
            for element, y_label in zip(X_test_SNN, y_test_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Test dataset")

            plt.figure(2)
            for element, y_label in zip(X_train_SNN, y_train_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Train dataset")

            plt.figure(3)
            for index, class_y in enumerate(predicted_y_test):
                color = "blue" if class_y <= 0 else "red"
                plt.scatter(X_test_SNN[index, class_no_1 - 1], X_test_SNN[index, class_no_2 - 1], color=color)
            plt.title("Results")

            plt.show()

if __name__ == "__main__":

    list_of_wavelets = ["db6"]
    main(dataset_type="wavelets", svm_go=False, snn_go=True, knn_go=False,
         number_of_hidden_neurons=9, list_of_wavelets=list_of_wavelets)



