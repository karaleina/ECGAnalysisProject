

def calculateTP(y_test = [], t_real = []):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for index, y_test_decicion in enumerate(y_test):
        if y_test_decicion == 1 and t_real[index] == 1:
            TP += 1

        elif y_test_decicion == 0 and t_real[index] == 1:
            FN += 1

        elif y_test_decicion == 0 and t_real[index] == 0:
            TN += 1

        elif y_test_decicion == 1 and t_real[index] == 0:
            FP += 1

    # print("TP", TP)
    # print("FP,", FP)
    # print("TP+FP", TP+FP, len(y_test))
    #
    # print("TN", TN)
    # print("FN,", FN)
    # print("TN+FN", TN + FN, len(y_test))

    sensitivity = TP / (FN + TP)
    specifity = TN / (TN + FP)

    # print("Czułość", sensitivity)
    # print("Specyficzność", specifity)

test = [1, 1, 1, 1, 1, 1, 1]
real = [0, 0, 0, 1, 0, 1, 0]

calculateTP(t_real=real, y_test=test)