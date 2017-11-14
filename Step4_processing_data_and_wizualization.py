from Step2_reading_and_correcting import read_with_pickle
from AF.tools.pca_tools import PCADimensionAnalyser
from matplotlib import pyplot as plt



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


if __name__ == "__main__":
    directory = "database/step3"
    X_test = read_with_pickle(directory + "/" + "X_test.pkl")
    X_train = read_with_pickle(directory + "/" + "X_train.pkl")
    X_test = transform_dataset_into_pcas_datasets(X_test)
