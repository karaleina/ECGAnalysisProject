from Step2_reading_and_correcting import read_with_pickle
from AF.tools.pca_tools import PCADimensionAnalyser
from matplotlib import pyplot as plt
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
    new_dataset = dataset.copy()
    for patient in dataset:
        list_rr_channel0 = dataset[patient]["channel0"]
        list_rr_channel1 = dataset[patient]["channel1"]
        for index in range(len(list_rr_channel0)):
            signal0 = list_rr_channel0[index].get_signal()
            signal1 = list_rr_channel0[index].get_signal()

            try:
                # Calculate coeffs
                dwt_a = DWTWaveletAnalyser()
                norm_coeff0 = dwt_a.get_wavelet_af_energy(signal0, frequency=128, wavelet=wavelet)
                norm_coeff1 = dwt_a.get_wavelet_af_energy(signal1, frequency=128, wavelet=wavelet)

                # Updating
                new_dataset[patient]["coeffs_pca"][index] = [norm_coeff0, norm_coeff1]

            except Exception:
                # TODO Co zrobic z tym, ze wtedy nie bedzie mial pacjent w tym zalamku coeffsow
                pass

if __name__ == "__main__":

    # PCA
    directory = "database/step3"
    X_test = read_with_pickle(directory + "/" + "X_test.pkl")
    X_train = read_with_pickle(directory + "/" + "X_train.pkl")

    # PCA
    X_test_pcas = transform_dataset_into_pcas_datasets(X_test)
    X_train_pcas = transform_dataset_into_pcas_datasets(X_train)

    # Wavelets
    X_test_wavelets_coeffs = transform_dataset_into_coeffs_dataset(X_test_pcas)
    X_train_wavelets_coeffs = transform_dataset_into_coeffs_dataset(X_train_pcas)

    # TODO testy NN z obecnym stanem prac
    # TODO falki
    # TODO emd
