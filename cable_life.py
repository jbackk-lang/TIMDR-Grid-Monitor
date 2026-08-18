"""
cable_life.py — szacowanie zużycia żywotności izolacji linii/kabla
elektroenergetycznego na podstawie profilu obciążenia. Działa dla
DOWOLNEGO kabla (miedź lub aluminium, PVC/XLPE/EPR) - materiał i typ
izolacji wpływają tylko na domyślne stałe (temperatura znamionowa,
projektowa żywotność), nie na kształt modelu.

Model uproszczony, oparty o dwie klasyczne, szeroko stosowane w
elektroenergetyce zasady inżynierskie:

1. **Kwadratowe grzanie rezystancyjne (I²R).** Przyrost temperatury
   przewodnika ponad otoczenie rośnie w przybliżeniu z KWADRATEM
   stosunku obciążenia do obciążenia znamionowego. To uproszczenie
   powszechnie stosowane przy orientacyjnym szacowaniu współczynników
   przeciążalności kabli - pokrewne (ale NIE identyczne, patrz niżej)
   pełnemu obwodowi cieplnemu z IEC 60287.
2. **Reguła Montsingera / Arrheniusa dla starzenia izolacji.**
   Żywotność izolacji polimerowej maleje w przybliżeniu WYKŁADNICZO z
   temperaturą pracy: żywotność skraca się o połowę na każde 8-10°C
   wzrostu temperatury ponad temperaturę znamionową (analogicznie do
   "reguły sześciu stopni" IEEE C57.91 dla transformatorów olejowych -
   tu z domyślnym krokiem 10°C, typowym dla izolacji polimerowej
   XLPE/PVC).

TO JEST PRZYBLIŻENIE INŻYNIERSKIE do orientacyjnej oceny TRENDU zużycia
żywotności pod danym profilem obciążenia - NIE certyfikowana kalkulacja
wg IEC 60287 (obciążalność prądowa długotrwała) ani IEC 60216 (badanie
wytrzymałości cieplnej izolacji). Rzeczywista żywotność zależy dodatkowo
od: rzeczywistej rezystancji cieplnej otoczenia (grunt/powietrze/kanał
kablowy, głębokość ułożenia, sąsiednie kable), wilgotności, LICZBY
CYKLI termicznych (nie tylko średniej temperatury - cykliczne
naprężenia mechaniczne izolacji przy grzaniu/chłodzeniu skracają
żywotność dodatkowo, czego ten model nie uwzględnia), jakości montażu,
uszkodzeń mechanicznych i jakości samego przewodnika. Dla oceny stanu
realnej linii skonsultuj się z uprawnionym elektroenergetykiem i/lub
wykonaj pomiar rezystancji izolacji (np. metodą VLF/tan-delta).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Typowe temperatury znamionowe przewodnika wg izolacji, zgodnie z
# konwencją IEC 60502 - wartości orientacyjne, sprawdź kartę katalogową
# konkretnego kabla dla realnej wartości.
RATED_TEMP_BY_INSULATION_C = {
    "pvc": 70.0,
    "xlpe": 90.0,
    "epr": 90.0,
}

# Orientacyjna projektowa żywotność izolacji przy pracy w temperaturze
# znamionowej - szeroko cytowane wartości rzędu dekad, różnią się
# między producentami/normami.
DEFAULT_DESIGN_LIFE_YEARS = {
    "pvc": 25.0,
    "xlpe": 30.0,
    "epr": 30.0,
}

VALID_MATERIALS = ("copper", "aluminum")


@dataclass
class CableSpec:
    rated_load_w: float
    conductor_material: str = "copper"
    insulation_type: str = "xlpe"
    ambient_temp_c: float = 25.0
    rated_conductor_temp_c: Optional[float] = None
    design_life_years: Optional[float] = None
    thermal_halving_deltaT_c: float = 10.0
    years_in_service: float = 0.0

    def __post_init__(self):
        if self.rated_load_w <= 0:
            raise ValueError("CableSpec: rated_load_w musi być dodatnie.")
        if self.years_in_service < 0:
            raise ValueError("CableSpec: years_in_service nie może być ujemne.")

        insulation = self.insulation_type.lower()
        if insulation not in RATED_TEMP_BY_INSULATION_C:
            raise ValueError(
                f"CableSpec: nieznany insulation_type '{self.insulation_type}'. "
                f"Dostępne: {sorted(RATED_TEMP_BY_INSULATION_C)}"
            )
        self.insulation_type = insulation

        material = self.conductor_material.lower()
        if material not in VALID_MATERIALS:
            raise ValueError(
                f"CableSpec: nieznany conductor_material '{self.conductor_material}'. "
                f"Dostępne: {sorted(VALID_MATERIALS)}"
            )
        self.conductor_material = material

        if self.rated_conductor_temp_c is None:
            self.rated_conductor_temp_c = RATED_TEMP_BY_INSULATION_C[insulation]
        if self.design_life_years is None:
            self.design_life_years = DEFAULT_DESIGN_LIFE_YEARS[insulation]

        if self.rated_conductor_temp_c <= self.ambient_temp_c:
            raise ValueError(
                "CableSpec: rated_conductor_temp_c musi być wyższa niż ambient_temp_c "
                f"(otrzymano rated={self.rated_conductor_temp_c}, ambient={self.ambient_temp_c})."
            )
        if self.thermal_halving_deltaT_c <= 0:
            raise ValueError("CableSpec: thermal_halving_deltaT_c musi być dodatnie.")
        if self.design_life_years <= 0:
            raise ValueError("CableSpec: design_life_years musi być dodatnie.")


def estimate_conductor_temp(load_w, spec: CableSpec) -> np.ndarray:
    """Szacowana temperatura przewodnika [°C] - I²R: przyrost ponad
    otoczenie skaluje się z kwadratem stosunku obciążenia do
    znamionowego."""
    load_w = np.asarray(load_w, dtype=float)
    load_ratio = load_w / spec.rated_load_w
    rated_rise = spec.rated_conductor_temp_c - spec.ambient_temp_c
    return spec.ambient_temp_c + rated_rise * load_ratio ** 2


# Górna granica sensowności współczynnika starzenia. Bez tego capu
# POJEDYNCZE próbki ekstremalnego przeciążenia (np. rated_load dobrane
# niżej niż realny profil obciążenia) potrafią wywindować wynik do
# absurdalnych, nieczytelnych wartości rzędu dziesiątek tysięcy - bo
# 2^(ΔT/10) jest FUNKCJĄ WYKŁADNICZĄ temperatury, więc uśrednianie po
# próbkach NIE spłaszcza ekstremów tak, jak zrobiłaby to średnia z
# wielkości liniowej (np. samej temperatury - stąd też widoczny w UI
# "paradoks": średnia temperatura może wyglądać umiarkowanie, a średni
# współczynnik starzenia i tak eksploduje, bo dominują go nieliczne
# gorące próbki). Powyżej ~1000x kabel w rzeczywistości dawno
# przekroczyłby dopuszczalną temperaturę, zadziałałoby zabezpieczenie
# (bezpiecznik/wyłącznik) albo izolacja uległaby fizycznemu zniszczeniu -
# większa liczba nie niesie już użytecznej informacji, tylko szum.
MAX_AGING_FACTOR = 1000.0


def estimate_aging_factor(conductor_temp_c, spec: CableSpec) -> np.ndarray:
    """Względne tempo zużycia żywotności izolacji (reguła Montsingera):
    1.0 = tempo projektowe (praca dokładnie w temperaturze znamionowej),
    <1.0 = wolniejsze starzenie (chłodniej niż znamionowo),
    >1.0 = szybsze starzenie (goręcej niż znamionowo). Ograniczone od
    góry do MAX_AGING_FACTOR - patrz komentarz przy stałej."""
    conductor_temp_c = np.asarray(conductor_temp_c, dtype=float)
    delta = conductor_temp_c - spec.rated_conductor_temp_c
    aging = 2.0 ** (delta / spec.thermal_halving_deltaT_c)
    return np.clip(aging, 0.0, MAX_AGING_FACTOR)


# Górna sensowna granica prognozy - powyżej tego model "wolniej niż
# projektowo starzejące się" przewiduje setki lat, co przestaje być
# użyteczne (niepewność modelu dominuje).
_MAX_SENSIBLE_PROJECTION_YEARS_FACTOR = 3.0


def estimate_remaining_life(load_w, sample_rate_hz: float, spec: CableSpec) -> dict:
    load_w = np.asarray(load_w, dtype=float)
    if len(load_w) == 0:
        raise ValueError("estimate_remaining_life: pusta seria obciążenia.")
    if sample_rate_hz <= 0:
        raise ValueError("estimate_remaining_life: sample_rate_hz musi być dodatnie.")

    temp = estimate_conductor_temp(load_w, spec)
    aging = estimate_aging_factor(temp, spec)
    mean_aging = float(np.mean(aging))
    mean_temp = float(np.mean(temp))

    window_years = (len(load_w) / sample_rate_hz) / (365.25 * 24 * 3600)
    equivalent_years_consumed_in_window = window_years * mean_aging

    remaining_design_years = max(0.0, spec.design_life_years - spec.years_in_service)
    cap = spec.design_life_years * _MAX_SENSIBLE_PROJECTION_YEARS_FACTOR
    if mean_aging > 1e-9:
        projected_remaining_years = min(remaining_design_years / mean_aging, cap)
    else:
        projected_remaining_years = cap

    if mean_aging >= 4.0:
        status = "KRYTYCZNE"
    elif mean_aging >= 1.5:
        status = "PRZYSPIESZONE_STARZENIE"
    else:
        status = "OK"

    return {
        "mean_conductor_temp_c": round(mean_temp, 1),
        "mean_aging_factor": round(mean_aging, 3),
        "window_years": round(window_years, 8),
        "equivalent_years_consumed_in_window": round(equivalent_years_consumed_in_window, 8),
        "years_in_service": spec.years_in_service,
        "design_life_years": spec.design_life_years,
        "estimated_remaining_years": round(projected_remaining_years, 2),
        "status": status,
        "rated_conductor_temp_c": spec.rated_conductor_temp_c,
        "ambient_temp_c": spec.ambient_temp_c,
        "insulation_type": spec.insulation_type,
        "conductor_material": spec.conductor_material,
    }
