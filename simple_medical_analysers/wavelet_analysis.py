import numpy as np


def calculate_wavelet_coeff_energy(coeff):

    return sum(np.power(coeff, 2))