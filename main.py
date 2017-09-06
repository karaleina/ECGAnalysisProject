from os import path
from analyzers import bothChannelsQRSDetector, rrIntervalsAnalyser

record = path.join("downloads", "04126")

analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
combined_rr = analyzerQRS.analyse(record, start_sample=0, stop_sample=10000, info=True, plotting=True)

analyzerRR = rrIntervalsAnalyser.rrIntervalsAnalyser(analyzerQRS)
list_of_intervals, rr_distances = analyzerRR.get_intervals(combined_rr, channel_no=1, time_margin=0.1)


# TODO Analiza odstępów RR
# (potrzebna miara opisująca rozkład histogramów?)
analyzerRR.plot_histogram(rr_distances, "Histogram odstępów RR")

# TODO Analiza falek
# (korzystająca z podzielonych fragmentów z RR?)
for interval in list_of_intervals:
    interval.plot_interval()

# TODO Baza referencyjna danych
# (skąd ją wziąć?)