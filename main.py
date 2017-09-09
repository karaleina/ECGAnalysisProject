from os import path
from analyzers import bothChannelsQRSDetector, rrIntervalsAnalyser, dimensionAnalyzer
from simple_medical_analysers import wavelet_analysis
import pywt
import numpy as np
from matplotlib import pyplot as plt


record = path.join("downloads", "04126")

analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
combined_rr = analyzerQRS.analyse(record, start_sample=0, stop_sample=10000, info=False, plotting=False)

analyzerRR = rrIntervalsAnalyser.rrIntervalsAnalyser(analyzerQRS)
list_of_intervals_chann0, rr_distances_chann0 = analyzerRR.get_intervals(combined_rr, channel_no=0, time_margin=0.1)
list_of_intervals_chann1, rr_distances_chann1 = analyzerRR.get_intervals(combined_rr, channel_no=1, time_margin=0.1)

# TODO Analiza odstępów RR
# (potrzebna miara opisująca rozkład histogramów?)
# analyzerRR.plot_histogram(rr_distances, "Histogram odstępów RR")

# TODO Analiza kształu (falki)
# (korzystająca z podzielonych fragmentów z RR?)

pca = dimensionAnalyzer.PCADimensionAnalyser()
ica = dimensionAnalyzer.ICADimensionAnalyzer()

for index, interval_chann0 in enumerate(list_of_intervals_chann0):

    interval_signal_chann0 = interval_chann0.get_signal()
    interval_signal_chann1 = list_of_intervals_chann1[index].get_signal()

    signal_chann_0_filtered = wavelet_analysis.filter_signal(interval_signal_chann0, wavelet="db6")
    signal_chann_1_filtered = wavelet_analysis.filter_signal(interval_signal_chann1, wavelet="db6")

    # PCA & ICA
    dataset = np.vstack((signal_chann_0_filtered, signal_chann_1_filtered)).T
    pca.calculate_new_dimension(dataset, pca_components=2)
    new_pca_dimension_dataset = pca.get_new_dimension(dataset).T

    ica.calculate_new_dimension(dataset, ica_components=2)
    new_ica_dimension_dataset = pca.get_new_dimension(dataset).T

    # plotting
    plt.figure(1)
    plt.subplot(3, 2, 1)
    plt.plot(interval_signal_chann0)
    plt.plot(interval_signal_chann1)
    plt.ylabel("EKG przed przetwarzaniem")

    plt.subplot(3, 2, 2)
    plt.plot(signal_chann_0_filtered)
    plt.plot(signal_chann_1_filtered)
    plt.ylabel("EKG po fltracji falkowej")

    plt.subplot(3, 2, 3)
    plt.plot(new_pca_dimension_dataset[0,:])
    plt.ylabel("Sygnał 1 po PCA")

    plt.subplot(3, 2, 4)
    plt.plot(new_pca_dimension_dataset[1, :])
    plt.ylabel("Sygnał 2 po PCA")

    plt.subplot(3, 2, 5)
    plt.plot(new_ica_dimension_dataset[0, :])
    plt.ylabel("Sygnał 2 po ICA")

    plt.subplot(3, 2, 6)
    plt.plot(new_ica_dimension_dataset[1, :])
    plt.ylabel("Sygnał 2 po ICA")
    plt.show()

#############################################################################################################3
# TODO Baza REFERENCYJNA danych

# (skąd ją wziąć?)


# TODO Czy podejście z analizą unormowanego współczynnika dekompozycji nie jest +/z równoważne z tranformatą Fouriera?..... (Nie tracimy informacji o położeniu?)

# TODO:
# TODO Wybrać odpowiednią tranfsormatę falkową
# TODO Wybrać drzewo dekompozycji opisujące najlepiej migotanie przedsionków

# TODO Dodać PCA i ICA i zobaczyć, jak to wpłynie na wynik.....
# TODO Zaprojektować sieć neuronową
