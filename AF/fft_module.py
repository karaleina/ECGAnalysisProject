from scipy.fftpack import fft
import numpy as np


def count_fft(signal, T_sampling):
    new_y = signal.ravel()

    # FFT
    N = len(new_y)

    yf = fft(new_y)
    xf = np.linspace(0.0, 1.0 / (2 * T_sampling), N // 2)

    yf = yf * 1.0 / len(yf) # normalized
    yf_abs = np.abs(yf[0:len(yf) // 2])

    return [xf, yf_abs]


def filter_fft(xf, yf_abs, f_min=-1, f_max=1000):
    filtred_y = yf_abs
    for index, (el_x, y_val) in enumerate(zip(xf, yf_abs)):
        if not f_min <= el_x <= f_max:
            filtred_y[index] = 0
    return filtred_y


def calculate_normalized_power_of_the_spectrum(yf_not_filtered, yf_filtered):
    return np.sum(np.power(yf_filtered, 2))/np.sum(np.power(yf_not_filtered, 2))


if __name__ == "__main__":
    a = [4,4]
    b = [5,4]
    result = calculate_normalized_power_of_the_spectrum(b, a)
    print(result)