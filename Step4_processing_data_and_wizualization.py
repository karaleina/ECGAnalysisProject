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
from AF import knn_algorithm



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


def transform_dataset_into_coeffs_dataset(dataset, *wavelets_names):
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
            for wavelet in wavelets_names:
                # Calculate coeffs
                dwt_a = DWTWaveletAnalyser()
                norm_coeff0 = dwt_a.get_wavelet_af_energy(signal0, frequency=128, wavelet=wavelet)
                norm_coeff1 = dwt_a.get_wavelet_af_energy(signal1, frequency=128, wavelet=wavelet)
                record_data.extend((norm_coeff0, norm_coeff1))
            # Updating
            y_info_dataset.append([diagnose, patient])
            X_coeffs_dataset.append(record_data)
    return np.array(X_coeffs_dataset), np.array(y_info_dataset)


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

    # TODO Douczanie samych przypadków trudnych:
    #  - z niezgodną przynależnością -  sieć neuronowa
    # - wytypowanie falek do ogolnej klasyfikacji i douczenia przypadkow trudnych

    # TODO FFT dataset
    # TODO EMD dataset

    directory = "database/step3"
    X_test = read_with_pickle(directory + "/" + "X_test.pkl")
    X_train = read_with_pickle(directory + "/" + "X_train.pkl")
    X_test_wavelets_coeffs, y_test_info = transform_dataset_into_coeffs_dataset(X_test, "db6", "db17", "dmey", "haar")
    X_train_wavelets_coeffs, y_train_info = transform_dataset_into_coeffs_dataset(X_train, "db6", "db17", "dmey", "haar")

    # ---------------- Wizualization parameters ----------------------
    class_no_1 = 3
    class_no_2 = 2

    # Datasets
    X_train_SNN = X_train_wavelets_coeffs
    X_train_SNN = X_train_SNN.astype('float')
    y_train_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_train_info[:, 0]]
    X_test_SNN = X_test_wavelets_coeffs
    X_test_SNN = X_test_SNN.astype('float')
    y_test_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_test_info[:, 0]]

    # -------------------------PARAMS------------------------------------

    svm_go = False
    snn_go = True
    knn_go = False
    data_visualisation = True

    # -------------------------Classification-----------------------------

    if svm_go is True:
        # ---------------------------- SVM -------------------------------------
        # Training
        clf = LinearSVC(random_state=0)
        clf.fit(X_train_SNN, y_train_SNN)
        dual_problem = False  # dual=False when n_samples > n_features.
        LinearSVC(C=1.0, class_weight=None, dual=dual_problem, fit_intercept=True,
                  intercept_scaling=1, loss='squared_hinge', max_iter=100000,
                  multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
                  verbose=0)

        # Testing
        predicted_y_test = clf.decision_function(X_test_SNN)
        predicted_y_test = [0 if element < 0 else 1 for element in predicted_y_test]
        print(predicted_y_test)

        quality = calculate_quality_of_classification(y_real=y_test_SNN, y_predictions=predicted_y_test)
        print("Specyficznosc:", quality["specifity"])
        print("Czułość:", quality["sensitivity"])

        if data_visualisation is True:
            # -------------- Data visualisation ---------------------------------
            for element, y_label in zip(X_test_wavelets_coeffs, y_test_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Test dataset")

            plt.figure(2)
            for element, y_label in zip(X_train_wavelets_coeffs, y_train_info):
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
        for x_test_instance in X_test_SNN:
            neighbours = knn_algorithm.getNeighbours(X_train_SNN, y_train_SNN, x_test_instance, k=1)
            predicted_y_value = knn_algorithm.getPrediction(neighbours, weighted_prediction=True)
            predicted_y_test.append(1) if predicted_y_value > 0.5 else predicted_y_test.append(0)

        quality = calculate_quality_of_classification(y_real=y_test_SNN, y_predictions=predicted_y_test)
        print("Specyficznosc:", quality["specifity"])
        print("Czułość:", quality["sensitivity"])

        if data_visualisation is True:
            # -------------- Data visualisation ---------------------------------
            for element, y_label in zip(X_test_wavelets_coeffs, y_test_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Test dataset")

            plt.figure(2)
            for element, y_label in zip(X_train_wavelets_coeffs, y_train_info):
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
    #----------------------------SNN--------------------------------------
        nn = SNN.NeuralNetwork([8, 16, 1])
        nn.fit(X_train_SNN, y_train_SNN)
        predicted_y_test = [1 if nn.predict(e) > 0.5 else 0 for e in X_test_SNN]

        quality = calculate_quality_of_classification(y_real=y_test_SNN, y_predictions=predicted_y_test)
        print("Specyficznosc:", quality["specifity"])
        print("Czułość:", quality["sensitivity"])

        if data_visualisation is True:
            # -------------- Data visualisation ---------------------------------
            for element, y_label in zip(X_test_wavelets_coeffs, y_test_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Test dataset")

            plt.figure(2)
            for element, y_label in zip(X_train_wavelets_coeffs, y_train_info):
                color = "blue" if y_label[0] == "ptb" else "red"
                plt.scatter(element[class_no_1 - 1], element[class_no_2 - 1], color=color)
            plt.title("Train dataset")

            plt.figure(3)
            for index, class_y in enumerate(predicted_y_test):
                color = "blue" if class_y <= 0 else "red"
                plt.scatter(X_test_SNN[index, class_no_1 - 1], X_test_SNN[index, class_no_2 - 1], color=color)
            plt.title("Results")

            plt.show()



