from os import path
from analyzers import bothChannelsQRSDetector, rrIntervalsAnalyser
from simple_medical_analysers import wavelet_analysis
import pywt
import numpy as np
from matplotlib import pyplot as plt


record = path.join("downloads", "04126")

analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
combined_rr = analyzerQRS.analyse(record, start_sample=0, stop_sample=10000, info=False, plotting=False)

analyzerRR = rrIntervalsAnalyser.rrIntervalsAnalyser(analyzerQRS)
list_of_intervals, rr_distances = analyzerRR.get_intervals(combined_rr, channel_no=1, time_margin=0.1)

# TODO Analiza odstępów RR
# (potrzebna miara opisująca rozkład histogramów?)
# analyzerRR.plot_histogram(rr_distances, "Histogram odstępów RR")

# TODO Analiza kształu (falki)
# (korzystająca z podzielonych fragmentów z RR?)

for interval in list_of_intervals:

    #interval_signal = [0.0, 2.0, 5.0, 6.0, 7.0, 9.0] # TESTOWY
    interval_signal = interval.get_signal()

    # Podejście 1
    #list_of_coeffs = pywt.wavedec(interval_signal, 'db6', level=4)
    # #signal_reconstructed = pywt.waverec(list_of_coeffs, 'db1')
    #
    # coeff_energies = []
    #
    # print(list_of_coeffs)
    #
    # for coeff in list_of_coeffs:
    #     coeff_energies.append(wavelet_analysis.calculate_wavelet_coeff_energy(coeff))
    #
    # print(coeff_energies)


    #Podejście 2
    #
    # for index, coeff in enumerate(list_of_coeffs):
    #     # if index >= len(list_of_coeffs) - 2:
    #     #     coeff = np.zeros(len(coeff))
    #     #     list_of_coeffs[index] = coeff
    #     if index != 0:
    #         coeff = np.zeros(len(coeff))
    #         list_of_coeffs[index] = coeff
    # signal_reconstructed = pywt.waverec(list_of_coeffs, 'db6')

    # Zerowanie nieuzywanych pasm (filtracja)

    # Podejście 2

    wavelet = "db6"
    (cA, cD) = pywt.dwt(interval_signal, wavelet)
    (cA2, cD2) = pywt.dwt(cA, wavelet)
    (cA3, cD3) = pywt.dwt(cA2, wavelet)
    (cA4, cD4) = pywt.dwt(cA3, wavelet)
    (cA5, cD5) = pywt.dwt(cA4, wavelet)
    (cA6, cD6) = pywt.dwt(cA5, wavelet)
    (cA7, cD7) = pywt.dwt(cA6, wavelet)

    cA7 = np.zeros(len(cA7))
    cD = np.zeros(len(cD))
    cD2 = np.zeros(len(cD2))

    list_of_coeffs = [cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD]
    signal_reconstructed = pywt.waverec(list_of_coeffs, wavelet)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(interval_signal)
    plt.ylabel("EKG przed rekonstukcją")
    plt.subplot(1,2,2)
    plt.plot(signal_reconstructed)
    plt.ylabel("EKG po rekonstrukcji")
    plt.show()

    # Podejście 2
    # (cA, cD) = pywt.dwt(interval_signal, 'db1')
    # (cA2, cD2) = pywt.dwt(cA, 'db1')
    # (cA3, cD3) = pywt.dwt(cA2, 'db1')
    # Na wejście sieci neuronowej będę podawać unormowaną energię dla danego współczynnika dekompozycji ?????

#############################################################################################################3
# TODO Baza REFERENCYJNA danych

# (skąd ją wziąć?)


# TODO Czy podejście z analizą unormowanego współczynnika dekompozycji nie jest +/z równoważne z tranformatą Fouriera?..... (Nie tracimy informacji o położeniu?)

# TODO:
# TODO Wybrać odpowiednią tranfsormatę falkową
# TODO Wybrać drzewo dekompozycji opisujące najlepiej migotanie przedsionków

# TODO Dodać PCA i ICA i zobaczyć, jak to wpłynie na wynik.....
# TODO Zaprojektować sieć neuronową
