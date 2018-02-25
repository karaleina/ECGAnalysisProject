from Step4_processing_data_and_wizualization import read_with_pickle, calculate_ROC_curve, transform_dataset_into_fft_power_dataset, transform_dataset_into_wavelets_coeffs_dataset

from matplotlib import pyplot as plt
from collections import OrderedDict
import os


nn = read_with_pickle(os.path.join(os.getcwd(), "results_ML", "nn" + str(2)) + ".pkl")
dataset_type = "fft"
list_of_wavelets = ["sym6", "db9", "coif10"]

data_visualisation = True
class_no_1 = 1
class_no_2 = 2


weights = nn.get_weights()
print("wagi", weights)
####
# READ DATASETS
# -----------------------CREATING DATASET---------------------------------------
directory = "database/step3"
X_test = read_with_pickle(directory + "/" + "X_test.pkl")
X_train = read_with_pickle(directory + "/" + "X_train.pkl")

plotting = False
if plotting is True:
    for patient in X_train:
        if X_train[patient]['diagnose'] == "aftdb":
            channel0 = X_train[patient]['channel0'][0].get_signal()
            channel1 = X_train[patient]['channel1'][0].get_signal()

            plt.figure(1).clf()
            plt.subplot(2, 1, 1)
            plt.plot(channel0)
            plt.ylabel("ECG 0", fontweight='bold')
            plt.title("Channel 0", fontweight='bold')
            plt.xlabel("n", fontweight='bold')
            plt.subplot(2, 1, 2)
            plt.plot(channel1)
            plt.title("Channel 1", fontweight='bold')
            plt.ylabel("ECG 1", fontweight='bold')
            plt.xlabel("n", fontweight='bold')
            plt.tight_layout()
            plt.show()

# ---------------------------WAVELET-DATASET----------------------------
if dataset_type == "wavelets":
    X_test_wavelets_coeffs, y_test_info = transform_dataset_into_wavelets_coeffs_dataset(X_test,
                                                                                         list_of_wavelets)  # , "dmey", "haar")
    X_train_wavelets_coeffs, y_train_info = transform_dataset_into_wavelets_coeffs_dataset(X_train,
                                                                                           list_of_wavelets)  # , "dmey", "haar")

    # Datasets
    X_train_SNN = X_train_wavelets_coeffs
    X_train_SNN = X_train_SNN.astype('float')
    y_train_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_train_info[:, 0]]
    X_test_SNN = X_test_wavelets_coeffs
    X_test_SNN = X_test_SNN.astype('float')
    y_test_SNN = [1 if y_label == "aftdb" else 0 for y_label in y_test_info[:, 0]]

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

predicted_y_test = [nn.predict(e).ravel() for e in X_test_SNN]
#predicted_y_train = [nn.predict(e).ravel() for e in X_train_SNN]


# PRINTING DATA FOR TABLE
for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    sens, one_minus_spec = calculate_ROC_curve(predicted_y_test_without_class=predicted_y_test,
                                               y_real=y_test_SNN,
                                               min_threshold=i,
                                               max_threshold=i, number_of_samples=1, plotting=False)
    print("KNN:", i, "sens:", sens, "1-spec", one_minus_spec)

# PLOTTING RESULTS
calculate_ROC_curve(predicted_y_test_without_class=predicted_y_test, y_real=y_test_SNN,
                                    min_threshold=0,
                                    max_threshold=1, number_of_samples=20, plotting=True)
# plt.show()
# plt.figure(1).clf()
#
# calculate_ROC_curve(predicted_y_test_without_class=predicted_y_train, y_real=y_train_SNN,
#                                     min_threshold=0,
#                                     max_threshold=1, number_of_samples=20, plotting=True)



if data_visualisation is True:
    # data_visuatlizate(X_test_SNN, y_test_info, predicted_y_test, y_train_info, X_train_SNN)

    plt.figure(3)
    for index, class_y in enumerate(predicted_y_test):
        color = "blue" if class_y <= 0.4 else "red"
        label = "ptb" if class_y <= 0.4 else "aftdb"
        plt.scatter(X_test_SNN[index, class_no_1 - 1], X_test_SNN[index, class_no_2 - 1], color=color,
                    label=label)
    plt.title("Results", fontweight="bold")
    plt.grid()
    plt.xlabel("coeff_1", fontweight="bold")
    plt.ylabel("coeff_2", fontweight="bold")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # plt.figure(4)
    # for index, class_y in enumerate(predicted_y_train):
    #     color = "blue" if class_y <= 0.4 else "red"
    #     label = "ptb" if class_y <= 0.4 else "aftdb"
    #     plt.scatter(X_train_SNN[index, class_no_1 - 1], X_train_SNN[index, class_no_2 - 1], color=color,
    #                 label=label)
    # plt.title("Results", fontweight="bold")
    # plt.grid()
    # plt.xlabel("coeff_1", fontweight="bold")
    # plt.ylabel("coeff_2", fontweight="bold")
    #
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())

plt.show()