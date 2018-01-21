from scipy.fftpack import fft
import numpy as np


class FFT_analyser(object):

    def __init__(self):
        pass

    @staticmethod
    def get_fft_coeff(signal, T_sampling, f_min, f_max):
        xf, yf_abs = FFT_analyser.count_fft(signal, T_sampling=T_sampling)
        yf_filtered = FFT_analyser.filter_fft(xf=xf, yf_abs=yf_abs, f_min=f_min, f_max=f_max)
        fft_coeff = FFT_analyser.calculate_normalized_power_of_the_spectrum(yf_not_filtered=yf_abs, yf_filtered=yf_filtered)
        return fft_coeff

    @staticmethod
    def count_fft(signal, T_sampling):
        new_y = signal.ravel()
        N = len(new_y)
        yf = fft(new_y)
        xf = np.linspace(0.0, 1.0 / (2 * T_sampling), N // 2)
        yf = yf * 1.0 / len(yf) # normalized
        yf_abs = np.abs(yf[0:len(yf) // 2])
        return [xf, yf_abs]

    @staticmethod
    def filter_fft(xf, yf_abs, f_min=-1, f_max=1000):
        filtred_y = np.empty_like(yf_abs)
        filtred_y[:] = yf_abs[:]
        for index, (el_x, y_val) in enumerate(zip(xf, yf_abs)):
            if not f_min <= el_x <= f_max:
                filtred_y[index] = 0
        return filtred_y

    @staticmethod
    def calculate_normalized_power_of_the_spectrum(yf_not_filtered, yf_filtered):
        return np.sum(np.power(yf_filtered, 2))/np.sum(np.power(yf_not_filtered, 2))


if __name__ == "__main__":
    a = [4, 4]
    b = [5, 4]

    result = calculate_normalized_power_of_the_spectrum(b, a)
    print(result)
