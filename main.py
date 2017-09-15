from os import path
from AF.analyzers import bothChannelsQRSDetector, rrIntervalsAnalyser, dimensionAnalyzer
from AF.simple_medical_analysers import wavelet_analysis
from AF.model import record
from matplotlib import pyplot as plt
import numpy as np


record_atrial_fibrillation = record.Record(path.join("downloads", "04126"))
record_atrial_fibrillation.set_frequency(250)

record_normal = record.Record(path.join("downloads", "nsrdb", "16420"))
record_normal.set_frequency(128)

# TODO RESAMPLING
# signal_af = record_atrial_fibrillation.get_signal()
# signal_norm = record_normal.get_signal()
#
# plt.subplot(2,1,1)
# plt.plot(signal_af)
#
# plt.subplot(2,1,2)
# plt.plot(signal_norm)
#
# plt.show()



for record in [record_normal, record_atrial_fibrillation]:

    print("---"*50)
    analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()

    combined_rr = analyzerQRS.analyse(record.get_path(), start_sample=0, stop_sample=1000, info=False, plotting=True, qrs_reading=True)

    analyzerRR = rrIntervalsAnalyser.rrIntervalsAnalyser(analyzerQRS)
    list_of_intervals_chann0, rr_distances_chann0 = analyzerRR.get_intervals(combined_rr, channel_no=0, time_margin=0.1)
    list_of_intervals_chann1, rr_distances_chann1 = analyzerRR.get_intervals(combined_rr, channel_no=1, time_margin=0.1)

    list_of_isoelectrics_chann0 = analyzerRR.get_list_of_intervals_isoelectric_line(list_of_intervals_chann0, margin_to_discard=0.4)
    list_of_isoelectrics_chann1 = analyzerRR.get_list_of_intervals_isoelectric_line(list_of_intervals_chann1, margin_to_discard=0.4)

    # TODO Analiza odstępów RR
    # (potrzebna miara opisująca rozkład histogramów?)
    # analyzerRR.plot_histogram(rr_distances_chann0, "Histogram odstępów RR")

    # TODO Analiza kształu (falki)
    # (korzystająca z podzielonych fragmentów z RR?)

    pca = dimensionAnalyzer.PCADimensionAnalyser()
    ica = dimensionAnalyzer.ICADimensionAnalyzer()

    for index, interval_chann0 in enumerate(list_of_intervals_chann0):

        interval_signal_chann0 = interval_chann0.get_signal()
        interval_signal_chann1 = list_of_intervals_chann1[index].get_signal()
        isoelectric_interval_chann0 = list_of_isoelectrics_chann0[index].get_signal()
        isoelectric_interval_chann1 = list_of_isoelectrics_chann1[index].get_signal()

        # Filtration
        signal_chann_0_filtered = wavelet_analysis.filter_signal(interval_signal_chann0, wavelet="db6")
        signal_chann_1_filtered = wavelet_analysis.filter_signal(interval_signal_chann1, wavelet="db6")
        signal_isoelectric_0_filtered = wavelet_analysis.filter_signal(isoelectric_interval_chann0, wavelet="db6")
        signal_isoelectric_1_filtered = wavelet_analysis.filter_signal(isoelectric_interval_chann1, wavelet="db6")

        # Wavelet Mayer
        coeffs0, AF_signal_0, old_signal = wavelet_analysis.get_AF_coeffs_and_AF_signal(signal_isoelectric_0_filtered, wavelet="dmey")
        coeffs1, AF_signal_1, old_signal = wavelet_analysis.get_AF_coeffs_and_AF_signal(signal_isoelectric_1_filtered, wavelet="dmey") #  [cd_3_5_Hz_0, cd_7_5_Hz_0, cd_12_Hz_0, AF_signal_0]

        print("Coeffs energies: " + str(wavelet_analysis.get_all_normed_coeffs_energies(old_signal, [coeffs0, coeffs1])))

        # PCA & ICA
        dataset = np.vstack((signal_isoelectric_0_filtered, signal_isoelectric_1_filtered)).T
        pca.calculate_new_dimension(dataset, pca_components=2)
        new_pca_dimension_dataset = pca.get_new_dimension(dataset).T

        ica.calculate_new_dimension(dataset, ica_components=2)
        new_ica_dimension_dataset = pca.get_new_dimension(dataset).T

        # plotting
        time_interval = 0.1 #s

        plt.ion()

        plt.figure(1).clear()
        plt.title("Różnica między sygnałem wejściowym, a wyciętą linią izoelektryczną")
        plt.subplot(2,1,1)
        plt.plot(signal_chann_0_filtered, label="kanał 1")
        plt.plot(signal_chann_1_filtered, label="kanał 2")
        plt.legend(loc="right")
        plt.subplot(2,1,2)
        plt.plot(signal_isoelectric_0_filtered, label="linia izoelekt. 1")
        plt.plot(signal_isoelectric_1_filtered, label='linia izoelekt. 2')
        plt.legend(loc="right")

        plt.figure(2).clear()
        plt.title("Sygnał AF linii izoelektrycznej otrzymany z falki Mayer'a")
        plt.subplot(2, 1, 1)
        plt.plot(AF_signal_0, label="Pierwszy kanał")
        plt.legend(loc="right")
        plt.subplot(2, 1, 2)
        plt.plot(AF_signal_1, label="Drugi kanał")
        plt.legend(loc="right")

        plt.figure(3).clear()
        plt.subplot(4, 2, 1)
        plt.plot(interval_signal_chann0)
        plt.plot(interval_signal_chann1)
        plt.ylabel("EKG przed przetwarzaniem")

        plt.subplot(4, 2, 2)
        plt.plot(signal_chann_0_filtered)
        plt.plot(signal_chann_1_filtered)
        plt.ylabel("EKG po filtracji falkowej")

        plt.subplot(4, 2, 3)
        plt.plot(isoelectric_interval_chann0)
        plt.plot(isoelectric_interval_chann1)
        plt.ylabel("Linia izoelektryczna przed filtracją")

        plt.subplot(4, 2, 4)
        plt.plot(signal_isoelectric_0_filtered)
        plt.plot(signal_isoelectric_1_filtered)
        plt.ylabel("Linia izoelektryczna po filtracji")

        plt.subplot(4, 2, 5)
        plt.plot(new_pca_dimension_dataset[0,:])
        plt.ylabel("Sygnał 1 po PCA")

        plt.subplot(4, 2, 6)
        plt.plot(new_pca_dimension_dataset[1, :])
        plt.ylabel("Sygnał 2 po PCA")

        plt.subplot(4, 2, 7)
        plt.plot(new_ica_dimension_dataset[0, :])
        plt.ylabel("Sygnał 2 po ICA")

        plt.subplot(4, 2, 8)
        plt.plot(new_ica_dimension_dataset[1, :])
        plt.ylabel("Sygnał 2 po ICA")

        plt.pause(time_interval)
    #
    # #############################################################################################################3
    # # TODO Baza REFERENCYJNA danych
    #
    # # (skąd ją wziąć?)
    #
    #
    # # TODO Czy podejście z analizą unormowanego współczynnika dekompozycji nie jest +/z równoważne z tranformatą Fouriera?..... (Nie tracimy informacji o położeniu?)
    #
    # # TODO:
    # # TODO Wybrać odpowiednią tranfsormatę falkową
    # # TODO Wybrać drzewo dekompozycji opisujące najlepiej migotanie przedsionków
    #
    # # TODO Dodać PCA i ICA i zobaczyć, jak to wpłynie na wynik.....
    # # TODO Zaprojektować sieć neuronową
