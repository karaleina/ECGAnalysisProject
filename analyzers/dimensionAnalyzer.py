import numpy as np
from sklearn import decomposition


class PCADimensionAnalyser(object):

    def __init__(self):
        self.__pca = None

    def calculate_new_dimension(self, train_data_matrix, pca_components = 1):

        train_data = np.mat(train_data_matrix)
        self.__pca = decomposition.PCA(n_components=pca_components).fit(train_data)

    def get_new_dimension(self, test_data_matrix):

        return self.__pca.transform(test_data_matrix)

class ICADimensionAnalyzer(object):

    def __init__(self):
        self.__ica = None

    def calculate_new_dimension(self, train_data_matrix, ica_components = 1):

        ica = decomposition.FastICA(n_components=ica_components)
        self.__ica = ica.fit(train_data_matrix)

    def get_new_dimension(self, test_data_matrix):
        self.__ica.transform(test_data_matrix)  # Estimate the sources