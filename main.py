from os import path
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser, dimensionAnalyzer, invertionAnalyzer
from AF.simple_medical_analysers import wavelet_analysis
from AF.model import record
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

# RECORDS

for index_str, str in enumerate(["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091", "05121", "08455"]):

    record_atrial_fibrillation = record.Record(path.join("downloads", "af", str), database="af")
    record_normal = record.Record(path.join("downloads", "ptb", "patient116", "s0302lre"), database="ptb")

    # PARAMS
    record_atrial_fibrillation.set_frequency(250)
    record_normal.set_frequency(1000)
    start = 0
    interval = 2.5

    # SIGNALS
    signals_af = record_atrial_fibrillation.get_signals(start, interval)
    signals_norm = record_normal.get_signals(start, interval, number_of_signals = 2)

    # RESAMPLING
    signals_norm = signal.resample(signals_norm, num=len(signals_af[:,0]))


    # channel_no = 0

    # for figure_idx in [1,2,3]:
    #     plt.figure(figure_idx)
    #     counter = 1
    #     for rows in [1, 2]:
    #         for columns in [1, 2]:
    #             plt.subplot(2, 2, counter)
    #             plt.plot(signals_norm[:,channel_no])
    #             plt.title("Signal ptb " + str(channel_no))
    #             counter += 1
    #             channel_no += 1
    #
    # plt.show()


    # # AUTOMATIC INVERTION ANALYSIS
    # signals_af = invertionAnalyzer.InvertionECGAnalyser.automatically_invert_signals(signals=signals_af)
    # signals_norm = invertionAnalyzer.InvertionECGAnalyser.automatically_invert_signals(signals=signals_norm)
    #
    # # MANUAL INVERTION
    # manual_invertion = True
    #
    # if manual_invertion == True:
    #     signals_af[:,0] = invertionAnalyzer.InvertionECGAnalyser.manually_invert_signal(signals_af[:,0])

    # PLOTTING SIGNALS


    plt.figure(index_str).clear()

    plt.subplot(2,2,1)
    plt.plot(signals_af[:, 0])
    plt.title("v1 af")

    plt.subplot(2,2,2)
    plt.plot(signals_af[:, 1])
    plt.title("v2 af")

    plt.subplot(2,2,3)
    plt.plot(signals_norm[:, 0])
    plt.title("v1 ptb")

    plt.subplot(2,2,4)
    plt.plot(signals_norm[:, 1])
    plt.title("v2 ptb")

plt.show()



    #
    #
    # for record in [record_normal, record_atrial_fibrillation]:
    #
    #     print("---"*50)
    #     analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
    #
    #     combined_rr = analyzerQRS.analyse(record.get_path(), start_sample=0, stop_sample=1000, info=False, plotting=True, qrs_reading=True)
    #
    #     analyzerRR = rrIntervalsAnalyser.rrIntervalsAnalyser(analyzerQRS)
    #     list_of_intervals_chann0, rr_distances_chann0 = analyzerRR.get_intervals(combined_rr, channel_no=0, time_margin=0.1)
    #     list_of_intervals_chann1, rr_distances_chann1 = analyzerRR.get_intervals(combined_rr, channel_no=1, time_margin=0.1)
    #
    #     list_of_isoelectrics_chann0 = analyzerRR.get_list_of_intervals_isoelectric_line(list_of_intervals_chann0, margin_to_discard=0.4)
    #     list_of_isoelectrics_chann1 = analyzerRR.get_list_of_intervals_isoelectric_line(list_of_intervals_chann1, margin_to_discard=0.4)
    #
    #     # TODO Analiza odstępów RR
    #     # (potrzebna miara opisująca rozkład histogramów?)
    #     # analyzerRR.plot_histogram(rr_distances_chann0, "Histogram odstępów RR")
    #
    #     # TODO Analiza kształu (falki)
    #     # (korzystająca z podzielonych fragmentów z RR?)
    #
    #     pca = dimensionAnalyzer.PCADimensionAnalyser()
    #     ica = dimensionAnalyzer.ICADimensionAnalyzer()
    #
    #     for index, interval_chann0 in enumerate(list_of_intervals_chann0):
    #
    #         interval_signal_chann0 = interval_chann0.get_signal()
    #         interval_signal_chann1 = list_of_intervals_chann1[index].get_signal()
    #         isoelectric_interval_chann0 = list_of_isoelectrics_chann0[index].get_signal()
    #         isoelectric_interval_chann1 = list_of_isoelectrics_chann1[index].get_signal()
    #
    #         # Filtration
    #         signal_chann_0_filtered = wavelet_analysis.filter_signal(interval_signal_chann0, wavelet="db6")
    #         signal_chann_1_filtered = wavelet_analysis.filter_signal(interval_signal_chann1, wavelet="db6")
    #         signal_isoelectric_0_filtered = wavelet_analysis.filter_signal(isoelectric_interval_chann0, wavelet="db6")
    #         signal_isoelectric_1_filtered = wavelet_analysis.filter_signal(isoelectric_interval_chann1, wavelet="db6")
    #
    #         # Wavelet Mayer
    #         coeffs0, AF_signal_0, old_signal = wavelet_analysis.get_AF_coeffs_and_AF_signal(signal_isoelectric_0_filtered, wavelet="dmey")
    #         coeffs1, AF_signal_1, old_signal = wavelet_analysis.get_AF_coeffs_and_AF_signal(signal_isoelectric_1_filtered, wavelet="dmey") #  [cd_3_5_Hz_0, cd_7_5_Hz_0, cd_12_Hz_0, AF_signal_0]
    #
    #         print("Coeffs energies: " + str(wavelet_analysis.get_all_normed_coeffs_energies(old_signal, [coeffs0, coeffs1])))
    #
    #         # PCA & ICA
    #         dataset = np.vstack((signal_isoelectric_0_filtered, signal_isoelectric_1_filtered)).T
    #         pca.calculate_new_dimension(dataset, pca_components=2)
    #         new_pca_dimension_dataset = pca.get_new_dimension(dataset).T
    #
    #         ica.calculate_new_dimension(dataset, ica_components=2)
    #         new_ica_dimension_dataset = pca.get_new_dimension(dataset).T
    #
    #         # plotting
    #         time_interval = 0.1 #s
    #
    #         plt.ion()
    #
    #         plt.figure(1).clear()
    #         plt.title("Różnica między sygnałem wejściowym, a wyciętą linią izoelektryczną")
    #         plt.subplot(2,1,1)
    #         plt.plot(signal_chann_0_filtered, label="kanał 1")
    #         plt.plot(signal_chann_1_filtered, label="kanał 2")
    #         plt.legend(loc="right")
    #         plt.subplot(2,1,2)

    #         plt.plot(signal_isoelectric_0_filtered, label="linia izoelekt. 1")
    #         plt.plot(signal_isoelectric_1_filtered, label='linia izoelekt. 2')
    #         plt.legend(loc="right")
    #
    #         plt.figure(2).clear()
    #         plt.title("Sygnał AF linii izoelektrycznej otrzymany z falki Mayer'a")
    #         plt.subplot(2, 1, 1)
    #         plt.plot(AF_signal_0, label="Pierwszy kanał")
    #         plt.legend(loc="right")
    #         plt.subplot(2, 1, 2)
    #         plt.plot(AF_signal_1, label="Drugi kanał")
    #         plt.legend(loc="right")
    #
    #         plt.figure(3).clear()
    #         plt.subplot(4, 2, 1)
    #         plt.plot(interval_signal_chann0)
    #         plt.plot(interval_signal_chann1)
    #         plt.ylabel("EKG przed przetwarzaniem")
    #
    #         plt.subplot(4, 2, 2)
    #         plt.plot(signal_chann_0_filtered)
    #         plt.plot(signal_chann_1_filtered)
    #         plt.ylabel("EKG po filtracji falkowej")
    #
    #         plt.subplot(4, 2, 3)
    #         plt.plot(isoelectric_interval_chann0)

    #         plt.plot(isoelectric_interval_chann1)
    #         plt.ylabel("Linia izoelektryczna przed filtracją")
    #
    #         plt.subplot(4, 2, 4)
    #         plt.plot(signal_isoelectric_0_filtered)
    #         plt.plot(signal_isoelectric_1_filtered)
    #         plt.ylabel("Linia izoelektryczna po filtracji")
    #
    #         plt.subplot(4, 2, 5)
    #         plt.plot(new_pca_dimension_dataset[0,:])
    #         plt.ylabel("Sygnał 1 po PCA")
    #
    #         plt.subplot(4, 2, 6)
    #         plt.plot(new_pca_dimension_dataset[1, :])
    #         plt.ylabel("Sygnał 2 po PCA")
    #
    #         plt.subplot(4, 2, 7)
    #         plt.plot(new_ica_dimension_dataset[0, :])
    #         plt.ylabel("Sygnał 2 po ICA")
    #
    #         plt.subplot(4, 2, 8)
    #         plt.plot(new_ica_dimension_dataset[1, :])
    #         plt.ylabel("Sygnał 2 po ICA")
    #
    #         plt.pause(time_interval)
    #     #
    #     # #############################################################################################################3
    #     # # TODO Baza REFERENCYJNA danych
    #     #
    #     # # (skąd ją wziąć?)
    #     #
    #     #
    #     # # TODO Czy podejście z analizą unormowanego współczynnika dekompozycji nie jest +/z równoważne z tranformatą Fouriera?..... (Nie tracimy informacji o położeniu?)
    #     #
    #     # # TODO:
    #     # # TODO Wybrać odpowiednią tranfsormatę falkową
    #     # # TODO Wybrać drzewo dekompozycji opisujące najlepiej migotanie przedsionków
    #     #
    #     # # TODO Dodać PCA i ICA i zobaczyć, jak to wpłynie na wynik.....
    #     # # TODO Zaprojektować sieć neuronową
