import numpy as np
from sklearn import decomposition


class PCADimensionAnalyser(object):

    def __init__(self):
        self.__pca = None

    def calculate_new_dimension(self, train_data_matrix, number_of_components=1):
        train_data = np.mat(train_data_matrix)
        self.__pca = decomposition.PCA(n_components=number_of_components).fit(train_data)

    def get_new_dimension(self, test_data_matrix):
        return self.__pca.transform(test_data_matrix)

    def calculate_pca_from_dataset(self, dataset):
        if dataset is None or len(dataset)==0:
            print("Pusty dataset")
            return None

        self.calculate_new_dimension(dataset.T)
        return self.get_new_dimension(dataset.T)

    def calculate_pcas(self, all):
        """ This method returns the new dimension dataset where
            1. dimension stands for piramis's index
            2. dimension stand for signal from particular piramid in thime
            """

        number_of_piramids = len(all[0, :, 0])
        number_of_frames = len(all[0, 0, :])
        new_dimension_dataset = np.zeros((number_of_piramids, number_of_frames))

        # Caluclate PCA for single piramid in LK algorithm
        for piramid_index in range(number_of_piramids):
            dataset = all[:, piramid_index, :]

            self.calculate_new_dimension(dataset.T)
            new_dimension_for_piramid = self.get_new_dimension(dataset.T)
            new_dimension_dataset[piramid_index, :] = new_dimension_for_piramid.ravel()

        return new_dimension_dataset