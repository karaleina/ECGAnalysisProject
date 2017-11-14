from Step2_reading_and_correcting import read_with_pickle, save_with_pickle


"""This code is creating test and train sets.
You need to run it after making any modifications to
aftdb_corrected or ptb pkl data"""


def element_is_proper_interval(potential_interval):
    try:
        potential_interval.get_signal()
        return True
    except AttributeError:
        return False


def add_test_and_train_datasets(train_dataset, test_dataset, database):
    """Creating TEST and TRAIN datasets as dictionaries
    with names of patients as keys"""

    for patient in database:
        list_rr_channel0 = database[patient]["channel0"]
        list_rr_channel1 = database[patient]["channel1"]

        train_dataset[patient] = {"channel0": [],
                            "channel1": [],
                            "diagnose": "AF"}

        test_dataset[patient] = {"channel0": [],
                           "channel1": [],
                           "diagnose": "AF"}

        for rr_index in range(len(list_rr_channel0)):
            if element_is_proper_interval(list_rr_channel0[rr_index]) \
                    and element_is_proper_interval(list_rr_channel0[rr_index]):

                if rr_index % 2 == 0:
                    train_dataset[patient]["channel0"].append(list_rr_channel0[rr_index])
                    train_dataset[patient]["channel1"].append(list_rr_channel1[rr_index])
                else:
                    test_dataset[patient]["channel0"].append(list_rr_channel0[rr_index])
                    test_dataset[patient]["channel1"].append(list_rr_channel1[rr_index])

    return train_dataset, test_dataset


if __name__ == "__main__":

    # Reading data and creating dataset
    aftdb = read_with_pickle("/home/karolina/PycharmProjects/atrialFibrillationAnalysisProject/database/step2/aftdb_corrected.pkl")
    ptb = read_with_pickle("/home/karolina/PycharmProjects/atrialFibrillationAnalysisProject/database/step1/ptb.pkl")

    X_train = {}
    X_test = {}

    for database in [aftdb, ptb]:
        X_train, X_test = add_test_and_train_datasets(
            train_dataset=X_train, test_dataset=X_test, database=database)

    save_with_pickle(X_train, "database/step3/X_train.pkl")
    save_with_pickle(X_test, "database/step3/X_test.pkl")