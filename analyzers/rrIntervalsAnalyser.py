from matplotlib import pyplot as plt

class rrIntervalsAnalyser(object):

    def __init__(self, bothChannelsQRSDetector):

        self._sampling_ratio = bothChannelsQRSDetector._sampling_ratio
        self._signals = bothChannelsQRSDetector._signals


    def rr_intervals_analyse(self, r_waves, channel_no=0, time_margin=0.2):

        samples_margin = time_margin * self._sampling_ratio
        channel = self._signals[:,channel_no]

        rr_distanses = []

        # TODO should return channel RRintervals
        prev_r_wave_index = -1
        current_rr_interval = None

        for r_wave_index in r_waves:
            if prev_r_wave_index > 0:
                current_rr_interval = channel[int(prev_r_wave_index - samples_margin):int(r_wave_index + samples_margin)]
                rr_distanses.append(int(r_wave_index - prev_r_wave_index))

                # TODO How to say which interval is with atrial fibrillation?
                # TODO Potrzebna referncyjna baza zdrowych ludzi...?

                ##### PLOTTING #####
                plt.figure(1)
                plt.plot(current_rr_interval)
                plt.show()
                ####################

                prev_r_wave_index = r_wave_index
            else:
                prev_r_wave_index = r_wave_index
                continue

        # print(rr_distanses)
        self.plot_histogram(rr_distanses, "Histogram odstępów RR")

        # TODO Miara opisująca rozkład histogramów

    def plot_histogram(self, signal, title):
        # TODO histogram (dedicated for rr_intervals)
        plt.hist(signal)
        plt.title(title)
        plt.show()
