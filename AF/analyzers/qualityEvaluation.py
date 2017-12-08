

def calculate_quality_of_classification(y_predictions = [], y_real = []):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for index, y_test_decicion in enumerate(y_predictions):
        if y_test_decicion == 1 and y_real[index] == 1:
            TP += 1

        elif y_test_decicion == 0 and y_real[index] == 1:
            FN += 1

        elif y_test_decicion == 0 and y_real[index] == 0:
            TN += 1

        elif y_test_decicion == 1 and y_real[index] == 0:
            FP += 1

    # print("TP", TP)
    # print("FP,", FP)
    # print("TP+FP", TP+FP, len(y_test))
    #
    # print("TN", TN)
    # print("FN,", FN)
    # print("TN+FN", TN+FN, len(y_test))

    sensitivity = TP / (FN + TP)
    specifity = TN / (TN + FP)

    # print("Czułość", sensitivity)
    # print("Specyficzność", specifity)
    #
    return {"sensitivity": sensitivity, "specifity": specifity}

if __name__ == "__main__":
    test = [1, 1, 1, 1, 0, 1, 0]
    real = [0, 0, 0, 1, 0, 1, 0]

    print(calculate_quality_of_classification(y_real=real, y_predictions=test))