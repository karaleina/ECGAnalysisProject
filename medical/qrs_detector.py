# -*- coding: utf-8 -*-


class QRSPanThompkinsDetector(object):
    def __init__(self):
        pass

    def detect_qrs(self, samples):


        # krok 1.
        # filtr dolnoprzepustowy
        # y(n) = y(n - 1) - y(n - 2) + x(n) - 2x(n - 6) + x(n - 12)
        dolny = [0] * len(samples)

        for n in range(12, len(samples)):  # liczenie sygnału po zastosowaniu filtru dolnego
            dolny[n] = dolny[n - 1] - dolny[n - 2] + samples[n] - 2 * samples[n - 6] + samples[
                n - 12]  # tutaj moja inwencja 1.000

        gorny = [0] * len(samples)  # sygnał po filtrze górnym
        for n in range(32, len(samples)):
            gorny[n] = dolny[n - 16] - (gorny[n - 1] + samples[n] - samples[n - 32]) / 32.

        pochodna_gorny = [0] * len(samples)  # miejsce na sygnał po zastosowaniu pochodnej
        for n in range(4, len(samples)):
            pochodna_gorny[n] = 0.125 * (2 * samples[n] + samples[n - 1] - samples[n - 3] - 2 * samples[n - 4])

        kwadrat = [0] * len(samples)  # po kwadratowaniu...
        for n in range(4, len(samples)):
            kwadrat[n] = pochodna_gorny[n] ** 2

        usredniony = [0] * len(samples)  # miejsce na sygnał po przejechaniu się oknem
        szerokosc_okna = 20
        for n in range(szerokosc_okna, len(samples)):
            usredniony[n] = sum(kwadrat[n - i] for i in range(szerokosc_okna)) / szerokosc_okna

        maks = max(usredniony)
        for n in range(len(samples)):  # normalizacja sygnału
            usredniony[n] = usredniony[n] / maks

        # rozpoznawanie stromych zboczy w ostatecznym sygnale
        zbocza = []
        poczatek_zbocza = 0
        for n in range(1, len(samples)):
            if usredniony[n] - usredniony[n - 1] > 0.01:  # zbocze trwa, idziemy dalej w prawo
                continue
            # zbocze się zakończyło, tzn. sygnał spadł w dół lub wzrost jest za mało stromy
            # jeśli długość zbocza jest co najmniej 5, to klasyfikujemy to jako zbocze.
            if n - poczatek_zbocza >= 5:  # jeśli zbocze jest dostatecznie długie
                zbocza.append((poczatek_zbocza, n - 1))
                # n - poczatek_zbocza
            poczatek_zbocza = n

        # teraz mamy zbocza, które w najbardziej typowym przypadku są jednego z dwóch rodzajów:
        # a) zbocze do punktu przegięia (tam gdzie jest załamek R)
        # b) zbocze od załamka R do pierwszego piku w sygnale
        # żeby się upewnić, że zachodzi ten przypadek, to koniec zbocza a) musi być blisko początku zbocza b)
        # jeśli tak jest to, twierdzę, że punkt R jest gdzieś pomiędzy końcem a) a początkiem b) -> więc mniemam na końcu a) bo tak ładniej działa

        start, koniec = zbocza[0]  # start, koniec oznaczają początek i koniec wcześniejsego zbocza
        zalamki_r = []
        for i in range(1, len(zbocza)):
            nowy_start, nowy_koniec = zbocza[i]  # nowy_start, nowy_koniec oznaczają początek i koniec następnego zbocza
            if nowy_start - koniec < 3:  # jeśli koniec a) jest blisko b)
                r = koniec
                zalamki_r.append((r, samples[r]))  # dodaję informację o załamku (czas, napięcie)
            start, koniec = nowy_start, nowy_koniec  # odtworzenie sytuacji wejściowej, następne zbocze staje się poprzednim

        punkty_q = []
        punkty_s = []

        for czas_r, wysokosc_r in zalamki_r:  # wyszukuje załamki q, s dookoła załamka r
            na_prawo = czas_r + 1
            while na_prawo < len(samples) - 1 and samples[czas_r] <= samples[na_prawo]:
                na_prawo += 1

            while na_prawo < len(samples) - 1 and samples[na_prawo] > samples[na_prawo + 1]:
                na_prawo += 1
            punkty_s.append((na_prawo, samples[na_prawo]))

            na_lewo = czas_r - 1
            while na_lewo > 0 and samples[czas_r] >= samples[na_lewo]:  # znalazł kandydata, najbliższy na lewo mniejszy od załamka r
                na_lewo -= 1

            while na_lewo > 0 and samples[na_lewo] > samples[na_lewo - 1]:  # schodzenie w dół
                na_lewo -= 1
            punkty_q.append((na_lewo, samples[na_lewo]))

        odstepy_rr = []
        for i in range(1, len(zalamki_r)):
            odstepy_rr.append(zalamki_r[i][0] - zalamki_r[i - 1][0])



        return zalamki_r #, punkty_q, punkty_s, odstepy_rr
