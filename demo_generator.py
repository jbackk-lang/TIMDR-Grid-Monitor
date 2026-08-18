"""
demo_generator.py — syntetyczne dane sieci energetycznej (230V/50Hz)
================================================================================
Generator realistycznych scenariuszy do demonstracji TIMDR-Grid-Monitor
bez podłączania realnego licznika/PLC (patrz też `csv_loader.py` -
import z pliku, i `device_client.py` - szkielet pod realne urządzenie).

Częstotliwość próbkowania: 100 Hz (10ms/próbka) - dobrana świadomie, nie
"na oko": to odpowiada konwencji "odświeżania półokresowego" (1/2-cycle
RMS refresh) z normy IEC 61000-4-30 używanej do oceny zdarzeń napięcia
(zapadów/zaników) - przy 50Hz jeden pełny okres to 20ms, więc pomiar co
10ms = dwa pomiary na okres. W realnych analizatorach jakości energii
różne parametry bywają odświeżane z różną częstotliwością (np. THD i
częstotliwość liczone z dłuższych, 10-minutowych okien wg EN 50160) -
tu, dla prostoty demo, wszystkie cztery kanały mają WSPÓLNĄ oś czasu.

Wartości nominalne: 230V / 50Hz (europejski standard niskiego napięcia).
"""

from __future__ import annotations

import numpy as np

from grid_monitor import TimdrEnergySignals

SAMPLE_RATE_HZ = 100.0
V_NOMINAL = 230.0
F_NOMINAL = 50.0
DEFAULT_RATED_LOAD_W = 10_000.0  # typowe przyłącze jednofazowe/mały obiekt


def _base_channels(n: int, rng: np.random.Generator, rated_load: float = DEFAULT_RATED_LOAD_W):
    """Zdrowa, 'cicha' sieć: napięcie/częstotliwość w normalnym paśmie
    EN 50160, THD niskie, obciążenie ze zmiennością dobową (30-50%
    mocy znamionowej), bez żadnych wstrzykniętych zdarzeń."""
    voltage = V_NOMINAL + rng.normal(0, 0.8, n)
    frequency = F_NOMINAL + rng.normal(0, 0.02, n)
    harmonics = np.abs(rng.normal(2.0, 0.4, n))  # THD% - zdrowa sieć, daleko od limitu 8%
    daily_pattern = 0.4 + 0.15 * np.sin(np.linspace(0, 4 * np.pi, n))
    load = rated_load * daily_pattern + rng.normal(0, rated_load * 0.01, n)
    return voltage, frequency, harmonics, load


def scenario_normalny(n: int = 6000, seed: int = 1, rated_load: float = DEFAULT_RATED_LOAD_W) -> TimdrEnergySignals:
    """Zdrowa sieć, bez zdarzeń - punkt odniesienia."""
    rng = np.random.default_rng(seed)
    voltage, frequency, harmonics, load = _base_channels(n, rng, rated_load)
    return TimdrEnergySignals(voltage=voltage, frequency=frequency, harmonics=harmonics, load=load)


def scenario_przeciazenie(n: int = 6000, seed: int = 2, rated_load: float = DEFAULT_RATED_LOAD_W) -> TimdrEnergySignals:
    """Przeciążenie: obciążenie rośnie powyżej progu ostrzegawczego
    (>90% mocy znamionowej) na dłuższy odcinek - typowy scenariusz
    "za dużo odbiorników na raz"."""
    rng = np.random.default_rng(seed)
    voltage, frequency, harmonics, load = _base_channels(n, rng, rated_load)
    start = n // 3
    duration = n // 6
    load[start:start + duration] = rated_load * (0.93 + 0.04 * rng.random(duration))
    # realny efekt uboczny przeciążenia - lekki spadek napięcia pod dużym obciążeniem
    voltage[start:start + duration] -= 6.0
    return TimdrEnergySignals(voltage=voltage, frequency=frequency, harmonics=harmonics, load=load)


def scenario_mikrozanik(n: int = 6000, seed: int = 3, rated_load: float = DEFAULT_RATED_LOAD_W) -> TimdrEnergySignals:
    """Mikro-zanik napięcia: krótki, głęboki spadek (np. zwarcie w
    sąsiedniej gałęzi sieci, przełączenie zasilania) - kilka zdarzeń o
    różnym czasie trwania."""
    rng = np.random.default_rng(seed)
    voltage, frequency, harmonics, load = _base_channels(n, rng, rated_load)
    events = [(n // 5, 3), (n // 2, 8), (int(n * 0.8), 2)]  # (start, czas trwania w próbkach)
    for start, dur in events:
        voltage[start:start + dur] = V_NOMINAL * 0.2 + rng.normal(0, 2, dur)
    return TimdrEnergySignals(voltage=voltage, frequency=frequency, harmonics=harmonics, load=load)


def scenario_anomalia_harmoniczna(n: int = 6000, seed: int = 4, rated_load: float = DEFAULT_RATED_LOAD_W) -> TimdrEnergySignals:
    """Anomalia harmoniczna: THD skacze powyżej limitu normy EN 50160
    (8%) - typowo od nieliniowego odbiornika (falownik, zasilacz
    impulsowy dużej mocy, ładowarka EV bez filtracji)."""
    rng = np.random.default_rng(seed)
    voltage, frequency, harmonics, load = _base_channels(n, rng, rated_load)
    start = n // 4
    duration = n // 8
    harmonics[start:start + duration] = 9.5 + rng.normal(0, 0.5, duration)
    return TimdrEnergySignals(voltage=voltage, frequency=frequency, harmonics=harmonics, load=load)


def scenario_cykliczne_zaklocenia(n: int = 6000, seed: int = 5, rated_load: float = DEFAULT_RATED_LOAD_W,
                                    period_samples: int = 150) -> TimdrEnergySignals:
    """Cykliczne zakłócenia: duży odbiornik cyklicznie się załącza
    (np. sprężarka, piec indukcyjny) - powtarzalne wahania obciążenia
    O REALNYM OKRESIE (nie losowy szum), w typowym zakresie sekund przy
    100Hz próbkowania (150 próbek = 1.5s)."""
    rng = np.random.default_rng(seed)
    voltage, frequency, harmonics, load = _base_channels(n, rng, rated_load)
    t = np.arange(n)
    cycle = np.where(np.mod(t, period_samples) < period_samples // 3, rated_load * 0.35, 0.0)
    load = load + cycle
    return TimdrEnergySignals(voltage=voltage, frequency=frequency, harmonics=harmonics, load=load)


def scenario_mieszany(n: int = 8000, seed: int = 6, rated_load: float = DEFAULT_RATED_LOAD_W) -> TimdrEnergySignals:
    """Kombinacja kilku zdarzeń w jednym oknie - test rezonansu
    międzykanałowego i realistycznego "złego dnia" na sieci."""
    rng = np.random.default_rng(seed)
    voltage, frequency, harmonics, load = _base_channels(n, rng, rated_load)

    # przeciążenie
    load[1000:1500] = rated_load * 0.94
    voltage[1000:1500] -= 5.0

    # mikro-zanik
    voltage[3000:3006] = V_NOMINAL * 0.15

    # anomalia harmoniczna jednoczesna z lekkim odchyleniem częstotliwości (rezonans)
    harmonics[5000:5300] = 9.8 + rng.normal(0, 0.4, 300)
    frequency[5000:5300] += 0.6

    # cykliczne wahania obciążenia w tle całego okna
    t = np.arange(n)
    load = load + np.where(np.mod(t, 200) < 60, rated_load * 0.15, 0.0)

    return TimdrEnergySignals(voltage=voltage, frequency=frequency, harmonics=harmonics, load=load)


SCENARIOS = {
    "normalny": scenario_normalny,
    "przeciazenie": scenario_przeciazenie,
    "mikrozanik": scenario_mikrozanik,
    "anomalia_harmoniczna": scenario_anomalia_harmoniczna,
    "cykliczne_zaklocenia": scenario_cykliczne_zaklocenia,
    "mieszany": scenario_mieszany,
}


def generate(scenario: str, **kwargs) -> TimdrEnergySignals:
    if scenario not in SCENARIOS:
        raise ValueError(f"Nieznany scenariusz '{scenario}'. Dostępne: {sorted(SCENARIOS)}")
    return SCENARIOS[scenario](**kwargs)


if __name__ == "__main__":
    for name in SCENARIOS:
        sig = generate(name)
        print(f"{name}: n={len(sig.load)}, V=[{sig.voltage.min():.1f},{sig.voltage.max():.1f}], "
              f"f=[{sig.frequency.min():.2f},{sig.frequency.max():.2f}], "
              f"THD=[{sig.harmonics.min():.1f},{sig.harmonics.max():.1f}], "
              f"load=[{sig.load.min():.0f},{sig.load.max():.0f}]")
