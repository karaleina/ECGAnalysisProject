# -*- coding: utf-8 -*-
import math


class QRSCompare(object):
   def __init__(self):
        pass

   def compare_segmentation(self, reference=None, test=None,
                            sampling_rate=250, tol_time=0.05):

        # Tol time to samples
        tol_time = sampling_rate * tol_time
        print("Samples_tolaration: " + str(tol_time))

        max_test = max(test)
        min_test = min(test)


        new_reference_for_TP_counting = []

        # Ograniczenie zbioru referencyjnego do potrzeb
        for element in reference:
            if min_test <= element <= max_test  and element >= min_test:
                new_reference_for_TP_counting.append(element)


        # 1 krok: znalezienie TP
        TP = []
        for element in new_reference_for_TP_counting:
            for item in test:
                potential_TP = None
                min_diff = None
                if item <= (element+tol_time) and item >= (element-tol_time):
                    potential_diff = math.fabs(item-element)
                    if min_diff == None:
                        min_diff = potential_diff
                        potential_TP = item
                    elif potential_diff <= min_diff:
                        min_diff = potential_diff
                        potential_TP = item
                if(potential_TP!=None): TP.append(potential_TP)


        # 2 krok: znalezienie FP
        FP = []
        for element in test:
            if not(element in TP):
                FP.append(element)


        # 3 krok: znalezienie FN
        FN = []
        for element in reference:
            is_potential_FN = True

            for item_test in test:
                if (element-tol_time) <= item_test <= (element+tol_time):
                    is_potential_FN = False
                    break

            if(is_potential_FN == True):
                FN.append(element)

        # 4 krok: znalezienie TN
        # ograniczam liczbę TN, aby punkty mało istotne nie zdominowały statystyki
        TN = []
        not_qrs = []

        for indeks, element in enumerate(reference):
            if indeks < (len(reference) - 1) :
                 not_qrs.append(reference[indeks] + (reference[indeks + 1] - reference[indeks]) / 2)



        for element in not_qrs:
            is_potential_TN = True
            for item in test:
                if item <= (element+tol_time) and item >= (element-tol_time):
                    is_potential_TN = False
                    break
            if is_potential_TN:
                TN.append(element)

        # print("TP: " + str(TP))
        # print("FP: " + str(FP))
        # print("FN: " + str(FN))
        # print("TN: " + str(TN))

        denom_sensiti = len(TP) + len(FN)
        if denom_sensiti != 0:
            sensivity = len(TP) / denom_sensiti
        else:
            sensivity = 0

        denom_specifi = (len(TN) + len(FP))
        if denom_specifi != 0:
            specifity = len(TN) / denom_specifi
        else:
            specifity = 0

        return [sensivity, specifity]
