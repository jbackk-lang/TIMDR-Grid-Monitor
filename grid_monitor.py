"""
grid_monitor.py — interpretacja sygnałów sieci energetycznej (TimdrEnergyMonitor)
================================================================================
Nazewnictwo klas (`TimdrEnergySignals`, `TimdrEnergyEvents`,
`TimdrEnergyMonitor`) pochodzi ze szkicu dostarczonego przez użytkownika.
Implementacja WEWNĄTRZ tych klas została napisana od nowa po empirycznej
weryfikacji szkicu - poniżej dokładnie co i dlaczego zmieniono (pełny
opis też w README.md, sekcja "Znalezione błędy"):

  1. `_detect_overloads` w szkicu liczył próg jako 90% z `max(load)` W TYM
     SAMYM ANALIZOWANYM OKNIE - samoreferencyjne: fałszywie flagowało
     ~21% próbek na zupełnie zdrowym profilu obciążenia (26% mocy
     znamionowej), bo próg zawsze "goni" lokalne maksimum, niezależnie od
     realnej pojemności linii/transformatora. Naprawiono: próg liczony
     jest teraz względem `rated_load` (moc znamionowa) - PARAMETRU
     WYMAGANEGO (jawny błąd, jeśli nie podany, nie cichy fallback na
     złamaną heurystykę).

  2. Kanał `frequency` w szkicu był w ogóle nieużywany w `analyze()` -
     mimo że monitoring częstotliwości był jednym z czterech jawnie
     wymaganych parametrów. Dodano `_detect_frequency_anomalies` wg
     normy EN 50160 (50Hz ±1% pasmo normalne, ±4% pasmo krytyczne dla
     systemów połączonych).

  3. `harmonics` w szkicu było niejednoznaczne (komentarz "widmo FFT" vs
     reszta klasy traktująca wszystko jako równoległe serie czasowe) i
     próg był WYŁĄCZNIE adaptacyjny (3× mediana okna) - zweryfikowano
     empirycznie, że to NIEBEZPIECZNE dla parametru z bezwzględnym,
     regulacyjnym limitem: na sieci przewlekle zanieczyszczonej (mediana
     THD=6%) próg adaptacyjny (18%) PRZEOCZAŁ realne przekroczenie normy
     EN 50160 (8%) wstrzyknięte jako THD=9.5%. Naprawiono: `harmonics`
     jest teraz jednoznacznie THD%(t) - jedna wartość na próbkę czasową,
     ta sama oś czasu co pozostałe kanały - z DWOMA niezależnymi
     warunkami wyzwalającymi (bezwzględny limit z normy LUB adaptacyjne
     odchylenie od lokalnej historii tej sieci - cokolwiek zadziała
     pierwsze, nigdy nie przeoczy regulacyjnego przekroczenia).

  4. `_detect_cyclic_disturbances` w szkicu robił autokorelację BEZ
     detrendu i bez filtrowania lokalnych maksimów (`np.argmax` po
     całym zakresie) - zweryfikowano empirycznie: na czystym, rosnącym
     trendzie obciążenia BEZ żadnej realnej okresowości, kod zgłaszał
     fałszywe "zakłócenie cykliczne" o okresie=cyclic_min_period (czysty
     artefakt granicy okna, nie prawdziwa struktura sygnału) - dokładnie
     ten sam bug "fałszywej periodyczności" znaleziony już wcześniej w
     kilku innych repo tego zestawu (bio_core.py, catalog_core.py,
     timdr_core_finance.py). Naprawiono: użyto `grid_core.rhythm()` -
     ten sam, zweryfikowany wzorzec (pełny detrend + znormalizowana
     autokorelacja per-lag + tylko ŚCIŚLE WEWNĘTRZNE lokalne maksima).
"""

from __future__ import annotations

import numpy as np

from grid_core import anomalies, defect, rhythm, resonance


class TimdrEnergySignals:
    """Cztery równoległe serie czasowe, TA SAMA częstotliwość próbkowania
    i ta sama długość - napięcie [V], częstotliwość [Hz], THD [%] (nie
    widmo - agregowany wskaźnik zniekształceń harmonicznych na próbkę
    czasową), obciążenie [W] (lub A - byle konsekwentnie w całym repo)."""

    def __init__(self, voltage, frequency, harmonics, load):
        self.voltage = np.asarray(voltage, dtype=float)
        self.frequency = np.asarray(frequency, dtype=float)
        self.harmonics = np.asarray(harmonics, dtype=float)
        self.load = np.asarray(load, dtype=float)

        lengths = {
            "voltage": len(self.voltage), "frequency": len(self.frequency),
            "harmonics": len(self.harmonics), "load": len(self.load),
        }
        if len(set(lengths.values())) > 1:
            # Nigdy cichy błąd przy niezgodności długości kanałów (lekcja
            # ze Synoptyka: schema mismatch = pułapka cichego zawodzenia).
            raise ValueError(f"TimdrEnergySignals: kanały mają różne długości: {lengths}")


class TimdrEnergyEvents:
    def __init__(self):
        self.overloads = []              # [(idx, load_value, rated_load)]
        self.micro_outages = []          # [(start_idx, duration_ms)]
        self.harmonic_anomalies = []     # [(idx, thd_value, powod)] powod: "limit_normy" | "odchylenie_adaptacyjne" | "oba"
        self.frequency_anomalies = []    # [(idx, freq_value, poziom)] poziom: "normalne_odchylenie" | "krytyczne"
        self.cyclic_disturbances = []    # [(channel, period_samples, power)]


class TimdrEnergyMonitor:
    """
    Parametry oparte na normie EN 50160 (charakterystyki napięcia
    zasilającego w publicznych sieciach dystrybucyjnych) tam, gdzie norma
    daje konkretne liczby - to NIE są dobrane "na oko" stałe, tylko
    powszechnie stosowane wartości referencyjne:
      - napięcie: ±10% wartości znamionowej (207-253V dla 230V) dla 95%
        10-minutowych średnich w tygodniu
      - częstotliwość (sieci połączone): ±1% (49.5-50.5Hz) normalne,
        +4%/-6% (47-52Hz) pasmo, które NIGDY nie powinno być przekroczone
      - THD napięcia: ≤ 8% (rzędy do 40. włącznie)
    """

    def __init__(
        self,
        v_nominal: float = 230.0,
        f_nominal: float = 50.0,
        rated_load: float | None = None,
        overload_threshold: float = 0.9,
        micro_outage_drop: float = 0.5,
        micro_outage_min_ms: float = 10,
        harmonic_thd_limit_pct: float = 8.0,
        harmonic_adaptive_factor: float = 3.0,
        freq_tolerance_pct: float = 1.0,
        freq_extended_tolerance_pct: float = 4.0,
        cyclic_min_period: int = 5,
        cyclic_max_lag: int = 200,
        cyclic_power_thresh: float = 0.35,
    ):
        self.v_nominal = v_nominal
        self.f_nominal = f_nominal
        self.rated_load = rated_load
        self.overload_threshold = overload_threshold
        self.micro_outage_drop = micro_outage_drop
        self.micro_outage_min_ms = micro_outage_min_ms
        self.harmonic_thd_limit_pct = harmonic_thd_limit_pct
        self.harmonic_adaptive_factor = harmonic_adaptive_factor
        self.freq_tolerance_pct = freq_tolerance_pct
        self.freq_extended_tolerance_pct = freq_extended_tolerance_pct
        self.cyclic_min_period = cyclic_min_period
        self.cyclic_max_lag = cyclic_max_lag
        self.cyclic_power_thresh = cyclic_power_thresh

    def analyze(self, signals: TimdrEnergySignals, sample_rate_hz: float) -> TimdrEnergyEvents:
        events = TimdrEnergyEvents()
        self._detect_overloads(signals, events)
        self._detect_micro_outages(signals, events, sample_rate_hz)
        self._detect_harmonic_anomalies(signals, events)
        self._detect_frequency_anomalies(signals, events)
        self._detect_cyclic_disturbances(signals, events)
        return events

    # -------------------------------------------------------------
    # Przeciążenia
    # -------------------------------------------------------------

    def _detect_overloads(self, signals: TimdrEnergySignals, events: TimdrEnergyEvents):
        if self.rated_load is None:
            # NIGDY cichy fallback na "90% z max okna" (błąd znaleziony w
            # kodzie referencyjnym) - jawny, czytelny błąd zamiast tego.
            raise ValueError(
                "TimdrEnergyMonitor: wykrywanie przeciążeń wymaga podania "
                "'rated_load' (moc znamionowa linii/transformatora w tych "
                "samych jednostkach co 'load') przy tworzeniu monitora - "
                "bez tego nie ma względem czego liczyć realnego przeciążenia."
            )
        load = signals.load
        threshold = self.overload_threshold * self.rated_load
        idx = np.where(load > threshold)[0]
        for i in idx:
            events.overloads.append((int(i), float(load[i]), float(self.rated_load)))

    # -------------------------------------------------------------
    # Mikro-zaniki napięcia
    # -------------------------------------------------------------

    def _detect_micro_outages(self, signals: TimdrEnergySignals, events: TimdrEnergyEvents, sample_rate_hz: float):
        voltage = signals.voltage
        threshold = self.micro_outage_drop * self.v_nominal
        below = voltage < threshold
        if not np.any(below):
            return

        indices = np.where(below)[0]
        start = indices[0]
        prev = indices[0]

        def _flush(s, p):
            duration_samples = p - s + 1
            duration_ms = 1000.0 * duration_samples / sample_rate_hz
            if duration_ms >= self.micro_outage_min_ms:
                events.micro_outages.append((int(s), float(duration_ms)))

        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                _flush(start, prev)
                start = idx
                prev = idx
        _flush(start, prev)

    # -------------------------------------------------------------
    # Anomalie harmoniczne (THD)
    # -------------------------------------------------------------

    def _detect_harmonic_anomalies(self, signals: TimdrEnergySignals, events: TimdrEnergyEvents):
        thd = signals.harmonics
        if len(thd) == 0:
            return

        over_limit = thd > self.harmonic_thd_limit_pct
        adaptive_idx = set(anomalies(thd, factor=self.harmonic_adaptive_factor).tolist())

        for i in range(len(thd)):
            is_over_limit = bool(over_limit[i])
            is_adaptive = i in adaptive_idx
            if not is_over_limit and not is_adaptive:
                continue
            if is_over_limit and is_adaptive:
                powod = "oba"
            elif is_over_limit:
                powod = "limit_normy"
            else:
                powod = "odchylenie_adaptacyjne"
            events.harmonic_anomalies.append((int(i), float(thd[i]), powod))

    # -------------------------------------------------------------
    # Anomalie częstotliwości
    # -------------------------------------------------------------

    def _detect_frequency_anomalies(self, signals: TimdrEnergySignals, events: TimdrEnergyEvents):
        freq = signals.frequency
        if len(freq) == 0:
            return

        normal_band = self.freq_tolerance_pct / 100.0 * self.f_nominal
        critical_band = self.freq_extended_tolerance_pct / 100.0 * self.f_nominal

        deviation = np.abs(freq - self.f_nominal)
        for i in range(len(freq)):
            if deviation[i] > critical_band:
                events.frequency_anomalies.append((int(i), float(freq[i]), "krytyczne"))
            elif deviation[i] > normal_band:
                events.frequency_anomalies.append((int(i), float(freq[i]), "normalne_odchylenie"))

    # -------------------------------------------------------------
    # Cykliczne zakłócenia
    # -------------------------------------------------------------

    def _detect_cyclic_disturbances(self, signals: TimdrEnergySignals, events: TimdrEnergyEvents):
        # (a) okresowość samego obciążenia (np. cykl dużego silnika/
        # sprężarki włączającego się co N próbek) - dokładnie to, co
        # miał na celu szkic użytkownika, ale teraz z naprawionym
        # detektorem okresowości (grid_core.rhythm, zob. docstring
        # modułu).
        periods_load, power_load = rhythm(
            signals.load, max_lag=self.cyclic_max_lag, power_thresh=self.cyclic_power_thresh,
        )
        for p in periods_load[:3]:
            events.cyclic_disturbances.append(("load", int(p), float(power_load)))

        # (b) okresowość samych ZDARZEŃ zakłócających (czy problemy
        # nawracają w regularnych odstępach, niezależnie od tego, czy
        # samo obciążenie jest okresowe) - bardziej dosłowna
        # interpretacja "cyklicznych zakłóceń" niż sama okresowość
        # surowego obciążenia (które może być normalnym, nieszkodliwym
        # wzorcem dobowym).
        disturbance_flags = np.zeros(len(signals.voltage))
        for i, _, _ in events.harmonic_anomalies:
            disturbance_flags[i] = 1.0
        for i, _, _ in events.frequency_anomalies:
            disturbance_flags[i] = 1.0
        if np.sum(disturbance_flags) >= 4:  # potrzeba paru zdarzeń, żeby okresowość miała sens
            periods_dist, power_dist = rhythm(
                disturbance_flags, max_lag=self.cyclic_max_lag, power_thresh=self.cyclic_power_thresh,
            )
            for p in periods_dist[:3]:
                events.cyclic_disturbances.append(("zdarzenia", int(p), float(power_dist)))
