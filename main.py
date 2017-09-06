from os import path
from analyzers import bothChannelsQRSDetector, rrIntervalsAnalyser

record = path.join("downloads", "04126")

analyzerQRS = bothChannelsQRSDetector.BothChannelsQRSDetector()
combined_rr = analyzerQRS.analyse(record, start_sample=0, stop_sample=10000, info=True, plotting=True)

analyzerRR = rrIntervalsAnalyser.rrIntervalsAnalyser(analyzerQRS)
analyzerRR.rr_intervals_analyse(combined_rr, channel_no=0)

# TODO Analiza odstępów RR
# (potrzebna miara opisująca rozkład histogramów?)

# TODO Analiza falek
# (korzystająca z podzielonych fragmentów z RR?)

# TODO Baza referencyjna danych
# (skąd ją wziąć?)