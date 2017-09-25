from AF.parsers import ecg_recording_parser_af
from matplotlib import pyplot as plt

from AF.simple_medical_analysers import wavelet_analysis
import numpy as np
from os import path
import wfdb


from os import path
from matplotlib import pyplot as plt
from AF.parsers import ecg_recording_parser_af

import wfdb
import numpy as np

file_with_all_names = path.join("downloads", "records_names_af")
file_object = open(file_with_all_names, "r")

afib_parsed_number = 0

for file_no in file_object:

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

    except FileNotFoundError:
        print("Brak kompletu plików dla rekordu o nr: " + str(file_no))

print("Liczba sparsowanych odcinków migotania przedsionków: ", afib_parsed_number)
