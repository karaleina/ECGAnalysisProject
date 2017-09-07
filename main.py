from os import path
from analyzers import bothChannelsQRSDetector, rrIntervalsAnalyser
from simple_medical_analysers import wavelet_analysis
import pywt
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
    list_of_coeffs = pywt.wavedec(interval_signal, 'db1', level=7)
    signal_reconstructed = pywt.waverec(list_of_coeffs, 'db1')

    coeff_energies = []

    for coeff in list_of_coeffs:
        coeff_energies.append(wavelet_analysis.calculate_wavelet_coeff_energy(coeff))

    print(coeff_energies)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(interval_signal)
    plt.subplot(1,2,2)
    plt.plot(signal_reconstructed)
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
