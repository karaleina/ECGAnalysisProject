
import numpy as np

class DetectionCombiner():

    def __init__(self):
        pass


    def verify(self, channelX=[], sampling_ratio=250, tol_compare_time=0.05):

        newChannelX = []
        tol_samples = tol_compare_time * sampling_ratio
        old_element = -1 - tol_samples

        for element in channelX:
            if element >= (old_element + tol_samples):
                newChannelX = np.append(newChannelX, element)

            old_element = element

        return newChannelX

    def combine(self, sampling_ratio, channel1=[], channel2=[], tol_compare_time=0.05):

        tol_samples = tol_compare_time * sampling_ratio

        new_qrs = channel1

        # Check for diffrerent qrs

        for element2 in channel2:

            element2IsSpecial = True

            for element1 in channel1:

                if element1 - tol_samples <= element2 <= element1 + tol_samples:
                    element2IsSpecial = False
                    break

            if element2IsSpecial:
                new_qrs = np.append(new_qrs, element2)

        return sorted(new_qrs)





