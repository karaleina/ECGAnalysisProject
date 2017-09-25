from AF.parsers import ecg_recording_parser_af
from matplotlib import pyplot as plt
from os import path
from matplotlib import pyplot as plt
import numpy as np
import pprint, pickle
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser
from AF.simple_medical_analysers import wavelet_analysis
import wfdb
from os import path
from AF.parsers import ecg_recording_parser_af


def get_list_of_intervals(signals_af):
    """Getting RR signals and returning list of intervals"""

    # Wavelet filtration
    numbers_of_signals = len(signals_af[0, :])
    numbers_of_samples = len(signals_af[:, 0])

    for i in range(numbers_of_signals):
        if numbers_of_samples % 2 == 1:
            signals_af = signals_af[:numbers_of_samples - 1, :]
        signals_af[:, i] = wavelet_analysis.filter_signal(signals_af[:, i], wavelet="db6", highcut=False)

    # Finding R-waves
    analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
    combined_rr = analyzerQRS.compare(signals_af, record_file=file_path, plotting=False, qrs_reading=False)

    # Creating list of isoelectrics lines
    rr_ia = RRIntervalsAnalyser.RRIntervalsAnalyser(analyzerQRS)
    list_of_intervals_chann0, rr_distances_chann0 = rr_ia.get_intervals(combined_rr, channel_no=0,
                                                                        time_margin=0.1)
    list_of_intervals_chann1, rr_distances_chann1 = rr_ia.get_intervals(combined_rr, channel_no=1,
                                                                        time_margin=0.1)

    return (list_of_intervals_chann0, list_of_intervals_chann1)

file_with_all_names = path.join("downloads", "records_names_af")
file_object = open(file_with_all_names, "r")

afib_parsed_number = 0

for file_no in file_object:
    # if file_no < "07162":
    #     continue

    dataset_per_file_no = {
                "channel0": [],
               "channel1": []}
    try:
        file_no = file_no.replace("\n", "")
        file_path = path.join("downloads", "af", file_no)
        annotations = wfdb.rdann(file_path, 'atr')

        annsamp = annotations.__dict__["annsamp"]
        names = annotations.__dict__["aux"]

        for index, (s, name) in enumerate(zip(annsamp, names)):
            print("Rekord nr: ", file_no, ", próbka: ", s, ", annotacja: ", name)
            if name == "(AFIB":
                start = s - 1000
                if s - 1000 < 0 :
                    start = s
                parser = ecg_recording_parser_af.ECGRecordingDataParser()

                signals_af = np.array(parser.parse(dat_file_name=file_path
                                    + ".dat", from_sample=start, to_sample=s+1000))
                afib_parsed_number += 1

                (list_of_intervals_chann0, list_of_intervals_chann1) = get_list_of_intervals(signals_af)

                # Uptadting dataset
                if len(list_of_intervals_chann1) == len(list_of_intervals_chann0):
                    # Saving isoelectric channel
                    dataset_per_file_no["channel0"] += list_of_intervals_chann0
                    dataset_per_file_no["channel1"] += list_of_intervals_chann1
                else:
                    print("SUMA KONTROLNA SIĘ NIE ZGADZA!!!!!!! :(((( ")

        output = open('database/af_data/' + file_no + '.pkl', 'wb')
        pickle.dump(dataset_per_file_no, output, -1)
        output.close()

    except FileNotFoundError:
        print("Brak kompletu plików dla rekordu o nr: " + str(file_no))

print("Liczba sparsowanych odcinków migotania przedsionków: ", afib_parsed_number)
