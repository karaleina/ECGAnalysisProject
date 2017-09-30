from os import path, makedirs
from matplotlib import pyplot as plt
import numpy as np
import pprint, pickle

from AF.parsers import ecg_recording_parser_af
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser
from AF.simple_medical_analysers import wavelet_analysis

pkl_file_no = "04043"
pkl_file = open('database/af_data/' + str(pkl_file_no) + '.pkl', 'rb')
dataset = pickle.load(pkl_file)
pkl_file.close()

new_pickle_dir = 'database/af_corrected_data/'
makedirs(path.dirname(new_pickle_dir), exist_ok=True)


filepath_new_pickle = path.join( new_pickle_dir, str(pkl_file_no) + '.pkl')
# filepath_new_pickle = path.abspath(filepath_new_pickle)

print(filepath_new_pickle)

#print(dataset)

plt.close("all")
print("len", len(dataset["channel1"]))


def start_looping(index=0):
    for i in range(len(dataset["channel1"])):
            if i >= index:
                plt.ion()
                plt.figure(i).clear()
                plt.plot(dataset["channel1"][i].get_signal())
                plt.plot(dataset["channel0"][i].get_signal())

                input_text = input(' write "delete" / "save" / or press enter to continue..')

                if input_text == "delete":
                    print("Zażądano usunięcia bieżącego załamka w pliku" + str(pkl_file))

                    # TODO
                    del dataset["channel1"][i]
                    del dataset["channel0"][i]

                    print("len", len(dataset["channel1"]))

                    start_looping(index=i)

                if input_text == "save":
                    print("Zażądano zapisania danych do pliku" + str(pkl_file))


                    output = open(filepath_new_pickle, 'wb')
                    pickle.dump(dataset, output, -1)
                    output.close()

                plt.pause(0.5)
                plt.close("all")
start_looping()

output = open(filepath_new_pickle, 'wb')
pickle.dump(dataset, output, -1)
output.close()


