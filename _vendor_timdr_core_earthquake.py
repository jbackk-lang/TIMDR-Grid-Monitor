"""
_vendor_timdr_core_earthquake.py -- ZWENDOROWANA (skopiowana 1:1, NIE
sibling-importowana) kopia TIMDR-Earthquake-Core/timdr_core_earthquake.py,
dnia 2026-09-10. flow()/twist()/trm()/anomalies() sa DOMENOWO NIEZALEZNE
(dzialaja na dowolnym jednokanalowym s(t)) -- ten adapter uzywal ich
wczesniej przez sys.path sibling-import z folderu-siostry
TIMDR-Earthquake-Core; zwendorowano, zeby to repo bylo samowystarczalne
po sklonowaniu SAMEGO SIEBIE, bez wymogu obecnosci innego repo na dysku
(decyzja podjeta na wyrazna prosbe: "repozytoria kodu maja byc niezalezne
od siebie"). To jest KOPIA, nie link -- ewentualne przyszle poprawki w
zrodlowym repo (TIMDR-Earthquake-Core) NIE propaguja sie tu automatycznie.

Oryginalny naglowek ponizej, bez zmian:
===========================================================================
TIMDR-Earthquake-Core — timdr_core_earthquake.py
==================================================
Rdzeń analizy sejsmicznej: lokalny gradient amplitudy (flow), nagłe
zmiany kierunku / początek wstrząsu (twist), wygładzenie szumu (TRM),
mikro-wstrząsy (anomalies) i punkty rozpoczęcia wstrząsu (fronts).

Wejście: t (znaczniki czasu, sekundy, ściśle rosnące), s (amplituda).
"""

import numpy as np
# NAPRAWIONE: `from scipy.signal import savgol_filter` na poziomie modulu
# wywalalo caly import tego pliku (a wiec i cale GUI) na maszynach, gdzie
# Device Guard/WDAC blokuje DLL-e scipy - nawet jesli uzytkownik wcale
# nie wybral metody "savgol" (domyslna w GUI to "median", ktora scipy w
# ogole nie potrzebuje). Import przeniesiony do wnetrza `_trm_savgol()`,
# ladowany tylko wtedy, gdy metoda "savgol" jest faktycznie uzyta.


class TIMDR_EarthquakeCore:
    """
    TIMDR Earthquake Detector Core
    - flow: lokalny gradient amplitudy sygnału sejsmicznego (LSQ względem czasu)
    - twist: nagłe zmiany kierunku (początek wstrząsu)
    - trm: wygładzenie szumu sejsmicznego (median / adaptive / savgol)
    - anomalies: mikro-wstrząsy
    - classify_anomalies: klasyfikacja ksztaltu anomalii (impuls/spike/step/drift/dropout)
    - fronts: punkt rozpoczęcia wstrząsu
    - sta_lta / trigger_onset: klasyczny picker STA/LTA (własna
      implementacja, zweryfikowana zgodność z ObsPy - patrz docstringi)
    - hybrid_trigger: STA/LTA potwierdzony przez TIMDR (twist + anomalie)
    """

    def __init__(self, k_neighbors=8, mad_scale=1.4826):
        self.k_neighbors = k_neighbors
        self.mad_scale = mad_scale

    def _safe_k(self, n):
        return min(self.k_neighbors, n)

    def _validate(self, t, s):
        t = np.asarray(t, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        if t.shape != s.shape:
            raise ValueError(f"t i s musza miec ten sam ksztalt, dostano {t.shape} i {s.shape}")
        if len(t) >= 2 and np.any(np.diff(t) <= 0):
            raise ValueError("t musi byc scisle rosnace (dt > 0 miedzy kolejnymi probkami)")
        return t, s

    # -----------------------------
    # Wspolny helper: k najblizszych sasiadow PO CZASIE, bez KDTree
    # -----------------------------
    def _nearest_k_bounds(self, t, i, k):
        """
        Zwraca (lo, hi) - wlaczne granice ciaglego wycinka t[lo:hi+1]
        zawierajacego k probek najblizszych t[i] w sensie |t[j]-t[i]|.

        Poprawka wydajnosciowa (bylo: KDTree na t.reshape(-1,1) per-punkt,
        z narzutem budowy drzewa + wywolania query() w petli Python).
        Poniewaz t jest SCISLE ROSNACE (wymuszone w _validate), k najblizszych
        po czasie zawsze tworzy CIAGLY przedzial indeksow wokol i - w prawo i
        w lewo odleglosc od t[i] rosnie monotonicznie, wiec nie trzeba
        przeszukiwac calej tablicy ani budowac drzewa: wystarczy dwuwskaznikowe
        rozszerzanie okna [lo, hi] o probke blizsza w danym kroku.

        WAZNE: to NIE jest to samo co sztywne okno po INDEKSIE [i-k:i+k].
        Przy nierownomiernym probkowaniu (przerwa w telemetrii, scalanie
        segmentow) okno po indeksie zlapaloby probki odlegle w czasie o
        sekundy zamiast faktycznie najblizszych - dokladnie ten blad, ktory
        `twist()` juz raz naprawil (patrz jego docstring). Ten helper liczy
        sasiedztwo po REALNEJ odleglosci w czasie `t`, wiec zachowuje sie
        identycznie jak KDTree na t.reshape(-1,1), tylko bez jego narzutu.
        """
        n = len(t)
        lo = hi = i
        while (hi - lo + 1) < k and (lo > 0 or hi < n - 1):
            left_ok = lo > 0
            right_ok = hi < n - 1
            if left_ok and (not right_ok or (t[i] - t[lo - 1]) <= (t[hi + 1] - t[i])):
                lo -= 1
            else:
                hi += 1
        return lo, hi

    # -----------------------------
    # FLOW — lokalny gradient LSQ
    # -----------------------------
    def flow(self, t, s):
        t, s = self._validate(t, s)
        n = len(t)
        if n < 3:
            return np.zeros_like(s)

        k = self._safe_k(n)
        grad = np.zeros_like(s)

        for i in range(n):
            lo, hi = self._nearest_k_bounds(t, i, k)
            tt = t[lo:hi + 1]
            ss = s[lo:hi + 1]

            A = np.column_stack([tt, np.ones_like(tt)])
            try:
                a, b = np.linalg.lstsq(A, ss, rcond=None)[0]
            except Exception:
                a = 0.0
            grad[i] = a

        return grad

    # -----------------------------
    # TWIST — nagłe zmiany kierunku
    # -----------------------------
    def twist(self, flow_grad, t, threshold=0.4):
        """
        POPRAWKA (bug krytyczny): oryginalny kod liczył
        `np.gradient(flow_grad)` BEZ przekazania `t` - traktując kolejne
        PRÓBKI jako równoodległe w czasie, mimo że `flow_grad` samo w
        sobie jest już poprawnie policzone względem rzeczywistego czasu
        w metodzie flow(). Dla danych z realną przerwą w rejestracji
        (typowe dla telemetrii sejsmicznej: dropout, przerwa między
        stacjami, scalanie segmentów) to daje SPURIOUS "twist" dokładnie
        na granicy przerwy.

        Zweryfikowano na czystej fali sinusoidalnej (0.1*sin(2*pi*2*t))
        z 3-sekundową przerwą w rejestracji pośrodku: oryginalny kod
        (gradient po indeksie) dawał na granicy przerwy szczyt "siły
        twistu" **3.8x większy** niż typowa (medianowa) wartość w reszcie
        sygnału - fałszywy alarm wywołany wyłącznie strukturą przerwy w
        danych, nie żadną prawdziwą zmianą fizyczną. Po poprawce
        (`np.gradient(flow_grad, t)`) granica przerwy wypada **0.0x**
        typowej wartości - poprawnie rozpoznana jako nic
        nadzwyczajnego.

        UWAGA (zmiana API): metoda teraz wymaga `t` jako argumentu -
        bez tego nie da się poprawnie liczyć gradientu względem czasu.
        """
        flow_grad = np.asarray(flow_grad, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        if len(flow_grad) < 3:
            return np.array([], dtype=int), np.zeros_like(flow_grad)
        if len(t) != len(flow_grad):
            raise ValueError("t i flow_grad musza miec ta sama dlugosc")

        dg = np.gradient(flow_grad, t)
        twist_strength = np.abs(dg)
        twist_points = np.where(twist_strength > threshold)[0]

        return twist_points, twist_strength

    # -----------------------------
    # TRM — wygładzenie (median / adaptive / savgol)
    # -----------------------------
    def trm(self, t, s, method="median", k_min=3, k_max=None,
             window_length=None, polyorder=3):
        """
        method="median" (domyslne, jak dotad): mediana z k najblizszych
        probek po czasie - k stale (self.k_neighbors).

        method="adaptive": jak wyzej, ale rozmiar okna k dopasowuje sie
        lokalnie do zmiennosci sygnalu - tam gdzie lokalna zmiennosc jest
        wysoka (blisko realnego wstrzasu/przejscia), okno jest MNIEJSZE, zeby
        mediana nie "usztywniala" prawdziwej zmiany; tam gdzie sygnal jest
        spokojny (szum tla), okno jest WIEKSZE, dla mocniejszego wygladzenia
        (rozwiazuje dokladnie problem opisany przy propozycji: stale k jest
        albo za sztywne w wysokiej amplitudzie, albo za miekkie w niskiej).
        k_min/k_max ograniczaja zakres adaptacji (domyslnie k_max = 3*k_neighbors).

        method="savgol": filtr Savitzky-Golay (scipy.signal.savgol_filter)
        jako alternatywa dla mediany - lepiej zachowuje ksztalt (np. strome
        zbocze wstrzasu) niz mediana, kosztem wiekszej wrazliwosci na
        pojedyncze odstajace probki. UWAGA: savgol dziala na indeksach, nie
        na `t` - zaklada w przyblizeniu rownomierne probkowanie (typowe dla
        pojedynczego kanalu sejsmicznego o stalej czestotliwosci probkowania;
        NIE uzywac przy nierownomiernych/scalanych seriach z lukami czasowymi -
        do tego sluzy method="median"/"adaptive", ktore licza sasiedztwo po
        rzeczywistym `t`).
        """
        t, s = self._validate(t, s)
        n = len(t)
        if n < 2:
            return s.copy()

        if method == "savgol":
            return self._trm_savgol(s, window_length, polyorder)
        if method == "adaptive":
            return self._trm_adaptive(t, s, k_min=k_min, k_max=k_max)
        if method != "median":
            raise ValueError(f"nieznana metoda trm: {method!r} (median/adaptive/savgol)")

        k = self._safe_k(n)
        smooth = np.zeros_like(s)
        for i in range(n):
            lo, hi = self._nearest_k_bounds(t, i, k)
            smooth[i] = np.median(s[lo:hi + 1])
        return smooth

    def _trm_adaptive(self, t, s, k_min=3, k_max=None):
        n = len(t)
        base_k = self._safe_k(n)
        if k_max is None:
            k_max = max(self.k_neighbors * 3, base_k)
        k_max = min(k_max, n)
        k_min = max(1, min(k_min, k_max))

        local_std = np.zeros(n)
        for i in range(n):
            lo, hi = self._nearest_k_bounds(t, i, base_k)
            local_std[i] = np.std(s[lo:hi + 1])

        std_max = np.max(local_std)
        norm = local_std / std_max if std_max > 1e-12 else np.zeros(n)
        # wysoka lokalna zmiennosc (norm~1) -> male okno (k_min);
        # niska lokalna zmiennosc (norm~0) -> duze okno (k_max)
        k_adapt = np.clip(
            np.round(k_max - norm * (k_max - k_min)).astype(int), k_min, k_max
        )

        smooth = np.zeros_like(s)
        for i in range(n):
            lo, hi = self._nearest_k_bounds(t, i, int(k_adapt[i]))
            smooth[i] = np.median(s[lo:hi + 1])
        return smooth

    def _trm_savgol(self, s, window_length, polyorder):
        try:
            from scipy.signal import savgol_filter
        except ImportError as e:
            raise ImportError(
                "Metoda 'savgol' wymaga scipy, ktore nie zaladowalo sie "
                f"w tym srodowisku ({e}). Uzyj metody 'median' lub "
                "'adaptive' (nie wymagaja scipy), albo napraw instalacje/"
                "polityke bezpieczenstwa blokujaca scipy."
            ) from e
        n = len(s)
        if window_length is None:
            window_length = min(n if n % 2 == 1 else n - 1, max(polyorder + 2, 11))
            if window_length % 2 == 0:
                window_length -= 1
        if window_length < polyorder + 2:
            raise ValueError(
                f"window_length ({window_length}) musi byc >= polyorder+2 ({polyorder + 2})"
            )
        if window_length > n:
            raise ValueError(f"window_length ({window_length}) wiekszy niz dlugosc sygnalu ({n})")
        return savgol_filter(s, window_length=window_length, polyorder=polyorder)

    # -----------------------------
    # ANOMALIE — mikro-wstrząsy
    # -----------------------------
    def anomalies(self, t, s, factor=3.0):
        """
        POPRAWKA (edge case): gdy sygnał jest silnie skwantowany /
        zawiera dużo powtórzonych wartości (typowe dla realnych danych z
        przetworników o ograniczonej rozdzielczości, albo skompresowanych
        formatów), mediana |reszt| (MAD) może wyjść dokładnie 0 - próg
        wychodzi wtedy 0, i KAŻDA niezerowa reszta (włącznie z szumem
        numerycznym) zostaje sklasyfikowana jako "anomalia".
        Zweryfikowano: na gładkim, tylko-zaokrąglonym sygnale (bez
        żadnej realnej anomalii) dawało to 7/30 (23%) fałszywych alarmów.
        Naprawiono: gdy MAD wychodzi ~0, próg spada na odch. std. reszt
        (a jeśli i to jest ~0 - sygnał faktycznie stały - próg to mała
        stała, nie zero).
        """
        t, s = self._validate(t, s)
        if len(t) == 0:
            return np.array([], dtype=int), np.array([]), 0.0
        smooth = self.trm(t, s)
        residuals = s - smooth

        mad = np.median(np.abs(residuals)) * self.mad_scale
        if mad <= 1e-12:
            std = np.std(residuals)
            mad = std if std > 1e-12 else 1e-9
        threshold = factor * mad

        anomaly_points = np.where(np.abs(residuals) > threshold)[0]

        return anomaly_points, residuals, threshold

    # -----------------------------
    # KLASYFIKACJA ANOMALII (ksztalt) — punkt 5
    # -----------------------------
    def classify_anomalies(self, t, s, factor=3.0, merge_gap=3, context=5,
                             revert_tol_factor=1.5, min_dropout_len=5,
                             dropout_eps=None, dropout_diff_frac=0.05):
        """
        Grupuje punkty z anomalies() w zdarzenia (sasiednie indeksy w
        odleglosci <= merge_gap sa jednym zdarzeniem) i kwalifikuje KSZTALT
        kazdego zdarzenia do jednej z 5 kategorii opisowych — NIE fizycznej
        interpretacji sejsmologicznej (na to trzeba widma/glebokosci/
        wielu stacji, patrz dyskusja o punktach 1/3/6), tylko ksztaltu
        przebiegu wokol anomalii, co jest policzalne z samego s(t):

        - "impuls": pojedyncza probka, mocno odstaje, natychmiast wraca do
          poprzedniego poziomu.
        - "spike":  krotki wybuch (kilka probek), potem wraca do poziomu
          sprzed zdarzenia.
        - "step":   poziom PRZED i PO zdarzeniu rozni sie trwale (nie wraca),
          bez wyraznego stopniowego narastania w trakcie zdarzenia.
        - "drift":  jak step, ale zmiana narasta stopniowo w trakcie
          zdarzenia (monotoniczny trend), a nie skokiem na starcie.
        - "dropout": dlugi bieg (>= min_dropout_len probek) o KROKACH
          MIEDZY SASIEDNIMI PROBKAMI dużo mniejszych niz typowy krok szumu
          w reszcie sygnalu - typowe dla utknietego czujnika/telemetrii,
          nie realnego ruchu gruntu (ktory prawie zawsze ma jakis szum tla,
          wiec kolejne probki realnie sie od siebie roznia).

        Dropout wykrywany jest OSOBNO od reszty (przeszukanie biegow w
        calym s, nie tylko w punktach juz oznaczonych przez anomalies()) -
        z prostego powodu: gdy plaski bieg jest dluzszy niz okno TRM, jego
        SRODEK ma lokalna mediane rowna sobie samemu, wiec residuum ~0 i
        anomalies() go NIE lapie - lapie tylko brzegi biegu jako dwa
        osobne, pozornie niezwiazane zdarzenia. Biegi dropout, ktore
        pokrywaja sie z takimi zdarzeniami z anomalies(), zastepuja je w
        wyniku (zeby nie raportowac tej samej awarii dwa razy pod dwiema
        etykietami).

        UWAGA: prog "stalosci" (dropout_eps) porownuje kroki MIEDZY
        SASIEDNIMI probkami (nie odleglosc od poczatku biegu), skalowany
        wzgledem typowego |diff| w calym sygnale (dropout_diff_frac=0.05
        domyslnie). To celowo NIE jest test "dokladnie ta sama wartosc" -
        preprocessing typu detrend (patrz SeismicLoader) odejmuje globalny
        trend liniowy, wiec idealnie staly odczyt czujnika po detrendzie
        dostaje maly, ale STALY spadek/wzrost między kolejnymi probkami
        zamiast byc dokladnie identyczny. Test na krok miedzy sasiadami
        (a nie na odleglosc od pierwszej probki biegu) toleruje taki
        liniowy dryf, jednoczesnie zostajac duzo czulszym niz realny szum
        tla (ktorego typowy krok jest rzedy wielkosci wiekszy).

        Zwraca liste dictow: {start, end, duration, type, level_shift}.
        Progi (merge_gap, revert_tol_factor, min_dropout_len,
        dropout_diff_frac) to heurystyki - do skalibrowania na realnych
        danych z etykietami, tak jak kazdy inny prog w tym projekcie.
        """
        t, s = self._validate(t, s)
        n = len(t)

        # --- dropout: biegi o prawie stalym kroku miedzy sasiadami,
        # niezaleznie od MAD (patrz uwaga w docstringu wyzej) ---
        if dropout_eps is None:
            diffs_all = np.abs(np.diff(s))
            typical_diff = np.median(diffs_all) if len(diffs_all) else 0.0
            if typical_diff <= 1e-12:
                std_s = np.std(s)
                typical_diff = std_s if std_s > 1e-12 else 1e-9
            dropout_eps = max(dropout_diff_frac * typical_diff, 1e-12)

        dropout_runs = []
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(s[j + 1] - s[j]) <= dropout_eps:
                j += 1
            if (j - i + 1) >= min_dropout_len:
                dropout_runs.append((i, j))
            i = j + 1

        anomaly_points, residuals, threshold = self.anomalies(t, s, factor=factor)

        labeled = []
        for a, b in dropout_runs:
            labeled.append({
                "start": a, "end": b, "duration": b - a + 1,
                "type": "dropout", "level_shift": 0.0,
            })

        if len(anomaly_points) == 0:
            return sorted(labeled, key=lambda e: e["start"])

        smooth = self.trm(t, s)
        events = []
        start = prev = int(anomaly_points[0])
        for idx in anomaly_points[1:]:
            idx = int(idx)
            if idx - prev <= merge_gap:
                prev = idx
                continue
            events.append((start, prev))
            start = prev = idx
        events.append((start, prev))

        global_std = np.std(residuals)
        global_std = global_std if global_std > 1e-12 else 1e-9
        revert_tol = revert_tol_factor * global_std

        def overlaps_dropout(a, b):
            return any(a <= dr_b and b >= dr_a for dr_a, dr_b in dropout_runs)

        for a, b in events:
            if overlaps_dropout(a, b):
                continue  # juz opisane jako dropout, nie duplikuj
            dur = b - a + 1
            pre_lo = max(0, a - context)
            post_hi = min(n, b + 1 + context)
            before = np.median(smooth[pre_lo:a]) if a > pre_lo else smooth[a]
            after = np.median(smooth[b + 1:post_hi]) if post_hi > b + 1 else smooth[b]
            level_shift = after - before
            seg = s[a:b + 1]
            reverts = abs(level_shift) <= revert_tol

            if dur == 1:
                label = "impuls" if reverts else "step"
            elif reverts:
                label = "spike"
            else:
                third = max(1, dur // 3)
                early = np.median(seg[:third])
                late = np.median(seg[-third:])
                gradual = abs(level_shift) > 1e-12 and abs(late - early) > 0.5 * abs(level_shift)
                label = "drift" if (gradual and dur >= 6) else "step"

            labeled.append({
                "start": a, "end": b, "duration": dur,
                "type": label, "level_shift": float(level_shift),
            })

        return sorted(labeled, key=lambda e: e["start"])

    # -----------------------------
    # FRONTS — początek wstrząsu
    # -----------------------------
    def fronts(self, t, s, twist_threshold=0.4, anomaly_factor=3.0):
        t, s = self._validate(t, s)
        flow_grad = self.flow(t, s)
        twist_pts, twist_strength = self.twist(flow_grad, t, threshold=twist_threshold)
        anomalies, residuals, th = self.anomalies(t, s, factor=anomaly_factor)

        candidates = np.intersect1d(twist_pts, anomalies)

        if len(flow_grad) < 3:
            return np.array([], dtype=int), twist_strength, residuals

        flow_med = np.median(np.abs(flow_grad))

        strong_fronts = [
            i for i in candidates
            if abs(flow_grad[i]) > 2 * flow_med
        ]

        return np.array(strong_fronts, dtype=int), twist_strength, residuals

    # -----------------------------
    # STA/LTA — klasyczny picker sejsmiczny (własna implementacja)
    # -----------------------------
    def sta_lta(self, s, nsta, nlta):
        """
        Classic STA/LTA: stosunek krótkoterminowej (STA) do
        długoterminowej (LTA) średniej energii sygnału - najbardziej
        rozpowszechniony klasyczny picker w sejsmologii (m.in. tak
        działa `obspy.signal.trigger.classic_sta_lta`).

        nsta, nlta: dlugosc okna STA/LTA w PROBKACH (nie w sekundach -
        przelicz sam: nsta = int(sta_sekundy * fs)).

        To jest WŁASNA implementacja napisana od podstaw wg klasycznego,
        powszechnie znanego wzoru (nie skopiowana z ObsPy) - zweryfikowana
        przez bezpośrednie porównanie z `obspy.signal.trigger.classic_sta_lta`
        na tych samych danych (przykładowy strumień dołączony do ObsPy,
        stacja BW.RJOB): wynik identyczny co do ~1e-14 (precyzja float)
        na całej długości sygnału, dla kilku różnych kombinacji okien.
        Test: `test_sta_lta_zgodny_z_obspy` (pomijany automatycznie, jeśli
        ObsPy nie jest zainstalowane - nie jest to twarda zależność
        TIMDR-Earthquake-Core).

        Implementacja przez sumy kroczące (cumsum) - O(n), nie O(n*nlta).
        Pierwsze `nlta - 1` próbek (niepełne okno LTA) zwraca 0 - tak jak
        w implementacji referencyjnej: stosunek z niepełnego, wciąż
        "rozpędzającego się" okna nie ma sensu fizycznego (dla pierwszej
        próbki STA/LTA z okna 1-elementowego zawsze wychodzi dokładnie
        1.0, niezależnie od danych - fałszywie "neutralny" wynik).
        """
        s = np.asarray(s, dtype=np.float64)
        n = len(s)
        if nsta < 1 or nlta < 1:
            raise ValueError("nsta i nlta musza byc >= 1")
        if nlta > n:
            return np.zeros(n)

        sq = s ** 2
        csq = np.concatenate([[0.0], np.cumsum(sq)])

        idx = np.arange(n)
        sta_start = np.maximum(0, idx - nsta + 1)
        lta_start = np.maximum(0, idx - nlta + 1)
        sta = (csq[idx + 1] - csq[sta_start]) / (idx - sta_start + 1)
        lta = (csq[idx + 1] - csq[lta_start]) / (idx - lta_start + 1)

        lta_safe = np.where(lta > 1e-12, lta, 1e-12)
        ratio = sta / lta_safe
        if nlta > 1:
            ratio[:nlta - 1] = 0.0
        return ratio

    def trigger_onset(self, charfct, thr_on, thr_off):
        """
        Histereza włącz/wyłącz na charakterystycznej funkcji (np. wyniku
        `sta_lta()`): trigger włącza się, gdy wartość >= thr_on, i
        wyłącza, gdy spadnie < thr_off (thr_off powinno być mniejsze niż
        thr_on - stąd "histereza", zapobiega migotaniu triggera wokół
        pojedynczego progu). Zwraca Nx2 tablicę [start_idx, koniec_idx]
        (oba indeksy WŁĄCZNIE, zgodnie z konwencją ObsPy).

        Zweryfikowano zgodność z `obspy.signal.trigger.trigger_onset` na
        3 różnych kombinacjach progów (w tym przypadek z dwoma osobnymi
        zdarzeniami) - identyczne wyniki. Test: `test_trigger_onset_zgodny_z_obspy`.
        """
        charfct = np.asarray(charfct, dtype=np.float64)
        n = len(charfct)
        on = False
        onsets = []
        start = None
        for i in range(n):
            v = charfct[i]
            if not on and v >= thr_on:
                on = True
                start = i
            elif on and v < thr_off:
                on = False
                onsets.append((start, i - 1))
        if on:
            onsets.append((start, n - 1))
        if not onsets:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(onsets, dtype=np.int64)

    # -----------------------------
    # HYBRYDA TIMDR + STA/LTA — punkt 7
    # -----------------------------
    def hybrid_trigger(self, t, s, nsta, nlta, twist_threshold=0.4, anomaly_factor=3.0,
                         sta_lta_thr_on=1.5, sta_lta_thr_off=0.5, tolerance=5):
        """
        Kazde zdarzenie z trigger_onset(sta_lta(...)) jest POTWIERDZONE
        tylko gdy w jego sasiedztwie (+/- tolerance probek) wystepuje
        RÓWNIEŻ silny twist (> twist_threshold) ORAZ punkt z anomalies()
        - trzy niezalezne detektory musza sie zgodzic, zamiast polegac na
        samym progu energii STA/LTA (ktory reaguje na kazdy wzrost energii,
        wliczajac np. impulsowy szum nie majacy ani charakterystyki nagle
        zmiany kierunku, ani statystycznie odstajacej amplitudy wzgledem
        lokalnego tla).

        Zwraca (confirmed, rejected):
        - confirmed: Nx2 tablica [start,end] (jak trigger_onset), tylko
          potwierdzone zdarzenia.
        - rejected: lista dictow o odrzuconych kandydatach STA/LTA i
          POWODZIE odrzucenia (missing_twist / missing_anomaly) - przydatne
          do diagnozy, czy hybryda jest zbyt restrykcyjna.

        UWAGA: to jest heurystyka redukcji false-positives z pojedynczego
        detektora energii, NIE walidacja wzgledem katalogu prawdziwych
        zdarzen. Czy faktycznie poprawia precision/recall (a nie tylko
        zmniejsza liczbe alarmow kosztem realnych wykryc), trzeba sprawdzic
        na danych z etykietami - dokladnie tak jak kazdy inny sygnal w tym
        projekcie (patrz RAPORT_TIMDR_Finanse.md dla metodologii out-of-sample).
        """
        t, s = self._validate(t, s)
        n = len(t)

        flow_grad = self.flow(t, s)
        _, twist_strength = self.twist(flow_grad, t, threshold=twist_threshold)
        anomaly_points, _, _ = self.anomalies(t, s, factor=anomaly_factor)
        anomaly_set = set(int(i) for i in anomaly_points)

        ratio = self.sta_lta(s, nsta, nlta)
        raw_onsets = self.trigger_onset(ratio, sta_lta_thr_on, sta_lta_thr_off)

        confirmed, rejected = [], []
        for start, end in raw_onsets:
            lo = max(0, int(start) - tolerance)
            hi = min(n - 1, int(end) + tolerance)
            has_twist = bool(np.any(twist_strength[lo:hi + 1] > twist_threshold))
            has_anomaly = any(idx in anomaly_set for idx in range(lo, hi + 1))
            if has_twist and has_anomaly:
                confirmed.append((int(start), int(end)))
            else:
                rejected.append({
                    "start": int(start), "end": int(end),
                    "missing_twist": not has_twist,
                    "missing_anomaly": not has_anomaly,
                })

        confirmed_arr = (
            np.array(confirmed, dtype=np.int64) if confirmed else np.empty((0, 2), dtype=np.int64)
        )
        return confirmed_arr, rejected
