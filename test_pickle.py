from os import path, makedirs
from matplotlib import pyplot as plt
import numpy as np
import pprint, pickle

from AF.parsers import ecg_recording_parser_af
from AF.analyzers import bothChannelsQRSDetector, RRIntervalsAnalyser
from AF.simple_medical_analysers import wavelet_analysis

pkl_file_no = "08378"#"07910"


new_pickle_dir = 'database/af_corrected4_data/'
filepath_old_pickle = path.join(new_pickle_dir, 'new_' + str(pkl_file_no) + '.pkl')

pkl_file = open(filepath_old_pickle, 'rb')#open('database/af_data/' + str(pkl_file_no) + '.pkl', 'rb')
dataset = pickle.load(pkl_file)
pkl_file.close()

# filepath_new_pickle = path.abspath(filepath_new_pickle)

print(filepath_old_pickle)

#print(dataset)

plt.close("all")
print("len", len(dataset["channel1"]))


new_choosen_dataset = dataset.copy()
new_choosen_dataset["channel1"] = []
new_choosen_dataset["channel0"] = []


def start_looping(index=0):
    for i in range(len(dataset["channel1"])):

            if i >= 0:
                plt.ion()
                plt.figure(0).clear()
                plt.title(str(i))
                plt.plot(dataset["channel1"][i].get_signal())
                plt.plot(dataset["channel0"][i].get_signal())

                input_text = input(' write "delete" / "save" / or press enter to continue..')

                if input_text == "delete":
                    print("Zażądano usunięcia bieżącego załamka w pliku" + str(pkl_file))

                    # TODO
                    del dataset["channel1"][i]
                    del dataset["channel0"][i]

                    print("len", len(dataset["channel1"]))
                    print("i", i)
                    start_looping(index=i)

                if input_text == "save":
                    print("Zażądano zapisania danych do pliku" + str(pkl_file))

                    filepath_new_pickle = path.join(new_pickle_dir, "new_" + str(pkl_file_no) + '.pkl')
                    output = open(filepath_new_pickle, 'wb')
                    pickle.dump(new_choosen_dataset, output, -1)
                    output.close()

                if input_text == "y":
                    new_choosen_dataset["channel1"].append(dataset["channel1"][i])
                    new_choosen_dataset["channel0"].append(dataset["channel0"][i])
                    print("len of new dataset ", len(new_choosen_dataset["channel0"]))

                plt.pause(0.05)
start_looping(0)

filepath_new_pickle = path.join(new_pickle_dir, str(pkl_file_no) + '.pkl')
output = open(filepath_new_pickle, 'wb')
pickle.dump(new_choosen_dataset, output, -1)
output.close()


