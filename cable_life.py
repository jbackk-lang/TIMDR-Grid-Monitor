"""
cable_life.py — szacowanie zużycia żywotności izolacji linii/kabla
elektroenergetycznego na podstawie profilu obciążenia. Działa dla
DOWOLNEGO kabla (miedź lub aluminium, PVC/XLPE/EPR) - materiał i typ
izolacji wpływają tylko na domyślne stałe (temperatura znamionowa,
projektowa żywotność), nie na kształt modelu.

Model uproszczony, oparty o TRZY klasyczne, szeroko stosowane w
elektroenergetyce zasady inżynierskie:

1. **Kwadratowe grzanie rezystancyjne (I²R).** Przyrost temperatury
   przewodnika ponad otoczenie rośnie w przybliżeniu z KWADRATEM
   stosunku obciążenia do obciążenia znamionowego. To uproszczenie
   powszechnie stosowane przy orientacyjnym szacowaniu współczynników
   przeciążalności kabli - pokrewne (ale NIE identyczne, patrz niżej)
   pełnemu obwodowi cieplnemu z IEC 60287.
2. **Równanie Arrheniusa dla starzenia cieplnego izolacji (IEC 60216).**
   Żywotność izolacji polimerowej jest w przybliżeniu wykładnicza
   względem ODWROTNOŚCI temperatury BEZWZGLĘDNEJ (Kelwiny):
   `L(T) = A * exp(Ea / (k*T))` - to jest RZECZYWISTE prawo fizyczne
   stojące za normą IEC 60216 (badanie wytrzymałości cieplnej izolacji),
   nie tylko "orientacyjna heurystyka". Popularna "reguła Montsingera"
   ("żywotność maleje o połowę na każde 8-10°C") to jedynie LOKALNA,
   LINIOWA W CELSJUSZACH aproksymacja tego równania, ważna tylko blisko
   punktu, dla którego została skalibrowana - PIERWSZA WERSJA tego
   modułu używała właśnie tej uproszczonej postaci (`2^(ΔT/10)`), co przy
   dużych odchyleniach od temperatury znamionowej dawało coraz bardziej
   niedokładne wyniki i wymuszało sztuczny, ręcznie dobrany sufit
   (`MAX_AGING_FACTOR`), żeby liczby pozostały czytelne - patrz
   `estimate_aging_factor()` niżej po pełne wyjaśnienie naprawy.
3. **Temperatura rozkładu izolacji jako fizyczny (nie numeryczny) sufit.**
   Powyżej pewnej temperatury materiał izolacyjny ulega termicznemu
   rozkładowi/zwęgleniu niezależnie od tego, jak długo tam pozostaje -
   to REALNA granica fizyczna, nie artefakt wzoru. Patrz
   `DECOMPOSITION_TEMP_BY_INSULATION_C` niżej.

TO JEST PRZYBLIŻENIE INŻYNIERSKIE do orientacyjnej oceny TRENDU zużycia
żywotności pod danym profilem obciążenia - NIE certyfikowana kalkulacja
wg IEC 60287 (obciążalność prądowa długotrwała) ani pełne badanie wg
IEC 60216 (tam stałe A/Ea wyznacza się eksperymentalnie dla KONKRETNEGO
materiału, tu są tylko wyprowadzone z popularnej reguły "połowa żywotności
na X stopni", patrz zastrzeżenie przy `thermal_halving_deltaT_c`).
Rzeczywista żywotność zależy dodatkowo od: rzeczywistej rezystancji
cieplnej otoczenia (grunt/powietrze/kanał kablowy, głębokość ułożenia,
sąsiednie kable), wilgotności, LICZBY CYKLI termicznych (nie tylko
średniej temperatury - cykliczne naprężenia mechaniczne izolacji przy
grzaniu/chłodzeniu skracają żywotność dodatkowo, czego ten model nie
uwzględnia), jakości montażu, uszkodzeń mechanicznych i jakości samego
przewodnika. Dla oceny stanu realnej linii skonsultuj się z uprawnionym
elektroenergetykiem i/lub wykonaj pomiar rezystancji izolacji (np. metodą
VLF/tan-delta).
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

# Orientacyjna temperatura POCZĄTKU termicznego rozkładu izolacji -
# literaturowe, typowe wartości (rząd wielkości, nie specyfikacja
# konkretnego produktu - sprawdź kartę katalogową/MSDS producenta dla
# realnej wartości). PVC zaczyna się rozkładać/odbarwiać już wyraźnie
# poniżej temperatury zapłonu (~390°C) - degradacja termiczna i wydzielanie
# HCl zaczyna się w praktyce już od ~140-160°C. XLPE/EPR (usieciowany
# polietylen / guma etylenowo-propylenowa) są znacznie bardziej odporne
# cieplnie - typowy początek rozkładu rzędu 300-350°C. To JEST FIZYCZNA
# granica modelu (nie numeryczny sufit jak MAX_AGING_FACTOR niżej) -
# powyżej niej materiał ulega nieodwracalnemu uszkodzeniu niezależnie od
# tego, jak krótko tam przebywał (por. IEC 60949 - granice temperatury
# zwarciowej, których nie wolno przekroczyć NAWET CHWILOWO).
DECOMPOSITION_TEMP_BY_INSULATION_C = {
    "pvc": 160.0,
    "xlpe": 350.0,
    "epr": 350.0,
}

# Referencyjna temperatura otoczenia, dla ktorej definiowany jest przyrost
# temperatury przy obciazeniu znamionowym (rated_conductor_temp_c). Musi
# byc STALA, niezalezna od aktualnego `ambient_temp_c` w CableSpec -
# patrz komentarz w estimate_conductor_temp() nizej po pelne wyjasnienie
# bledu, ktory powstawal przy ich pomyleniu. 25 C to typowa referencyjna
# temperatura otoczenia w kartach katalogowych kabli (pokrewna, choc nie
# identyczna, referencyjnym warunkom z IEC 60287).
REFERENCE_AMBIENT_TEMP_C = 25.0


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
    decomposition_temp_c: Optional[float] = None

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
        if self.decomposition_temp_c is None:
            self.decomposition_temp_c = DECOMPOSITION_TEMP_BY_INSULATION_C[insulation]

        if self.rated_conductor_temp_c <= self.ambient_temp_c:
            raise ValueError(
                "CableSpec: rated_conductor_temp_c musi być wyższa niż ambient_temp_c "
                f"(otrzymano rated={self.rated_conductor_temp_c}, ambient={self.ambient_temp_c})."
            )
        if self.thermal_halving_deltaT_c <= 0:
            raise ValueError("CableSpec: thermal_halving_deltaT_c musi być dodatnie.")
        if self.design_life_years <= 0:
            raise ValueError("CableSpec: design_life_years musi być dodatnie.")
        if self.decomposition_temp_c <= self.rated_conductor_temp_c:
            raise ValueError(
                "CableSpec: decomposition_temp_c musi być wyższa niż rated_conductor_temp_c "
                f"(otrzymano decomposition={self.decomposition_temp_c}, "
                f"rated={self.rated_conductor_temp_c})."
            )


def estimate_conductor_temp(load_w, spec: CableSpec) -> np.ndarray:
    """Szacowana temperatura przewodnika [°C] - I²R: przyrost ponad
    otoczenie skaluje się z kwadratem stosunku obciążenia do
    znamionowego.

    ZNALEZIONY I NAPRAWIONY BŁĄD: pierwsza wersja liczyła
    `rated_rise = spec.rated_conductor_temp_c - spec.ambient_temp_c`, czyli
    przyrost temperatury WZGLĘDEM TEGO SAMEGO `ambient_temp_c`, które
    zaraz potem było z powrotem dodawane. Przy obciążeniu dokładnie
    znamionowym (load_ratio=1) dawało to zawsze
    `temp = ambient + (rated - ambient) = rated` - czyli wynik był
    STAŁY i CAŁKOWICIE NIEZALEŻNY od `ambient_temp_c`, niezależnie od
    tego, jak gorące/zimne otoczenie użytkownik wpisał. Przy przeciążeniu
    (load_ratio>1) było jeszcze gorzej: podniesienie ambient_temp_c
    OBNIŻAŁO wynikową temperaturę przewodnika - fizycznie odwrotny
    kierunek. Naprawa: przyrost temperatury przy obciążeniu znamionowym
    liczony jest teraz względem STAŁEJ referencyjnej temperatury otoczenia
    (`REFERENCE_AMBIENT_TEMP_C`, niezależnej od `spec.ambient_temp_c`) -
    dzięki temu podniesienie faktycznej temperatury otoczenia zawsze
    podnosi wynikową temperaturę przewodnika 1:1 (fizycznie poprawnie dla
    stałego źródła ciepła I²R nad zmiennym otoczeniem), niezależnie od
    poziomu obciążenia. Regresja: `test_temp_rosnie_z_ambient_przy_dowolnym_obciazeniu`
    w test_cable_life.py."""
    load_w = np.asarray(load_w, dtype=float)
    load_ratio = load_w / spec.rated_load_w
    rated_rise = spec.rated_conductor_temp_c - REFERENCE_AMBIENT_TEMP_C
    return spec.ambient_temp_c + rated_rise * load_ratio ** 2


# Przelicznik Celsjusz -> Kelwin (rownanie Arrheniusa jest liniowe w
# 1/T_bezwzgledne, NIE w temperaturze Celsjusza - patrz nizej).
_CELSIUS_TO_KELVIN_OFFSET = 273.15

# CZYSTO NUMERYCZNY sufit bezpieczenstwa (przed przepelnieniem float /
# nieczytelnymi wartosciami przy skrajnie blednych parametrach wejsciowych,
# np. rated_load_w ustawione absurdalnie nisko) - w PRZECIWIENSTWIE do
# poprzedniej wersji tego modulu, NIE jest to juz glowny "bezpiecznik"
# przy realistycznym przegrzaniu. Te role pelni teraz decomposition_temp_c
# (fizyczna granica materialu, patrz estimate_remaining_life() nizej) -
# dlatego ten numeryczny sufit jest ustawiony celowo bardzo wysoko (rzadko
# powinien sie w ogole uaktywnic przy sensownych parametrach).
MAX_AGING_FACTOR = 1.0e6


def estimate_aging_factor(conductor_temp_c, spec: CableSpec) -> np.ndarray:
    """Względne tempo zużycia żywotności izolacji wg równania Arrheniusa
    (IEC 60216, NIE uproszczonej reguły Montsingera - patrz docstring
    modułu): 1.0 = tempo projektowe (praca dokładnie w temperaturze
    znamionowej), <1.0 = wolniejsze starzenie (chłodniej niż znamionowo),
    >1.0 = szybsze starzenie (goręcej niż znamionowo).

    ZNALEZIONY I NAPRAWIONY BŁĄD: pierwsza wersja liczyła
    `2.0 ** (delta_c / thermal_halving_deltaT_c)` - funkcję wykładniczą
    LINIOWĄ WZGLĘDEM RÓŻNICY TEMPERATUR W CELSJUSZACH (reguła
    Montsingera). To jest tylko lokalna aproksymacja prawdziwego prawa
    Arrheniusa (wykładnicza względem ODWROTNOŚCI temperatury
    BEZWZGLĘDNEJ), ważna jedynie blisko punktu kalibracji - przy dużych
    odchyleniach systematycznie się myli. W szczególności dawała
    SYMETRYCZNE podwojenie/połowienie żywotności dla +ΔT/-ΔT, podczas gdy
    prawdziwa fizyka jest ASYMETRYCZNA: schłodzenie o ΔT daje WIĘKSZY
    zysk żywotności niż podgrzanie o ΔT ją kosztuje (konsekwencja
    krzywizny funkcji 1/T). Przy skrajnym przegrzaniu rosła też bez
    ograniczeń, co wymuszało sztuczny, nisko ustawiony sufit
    (MAX_AGING_FACTOR=1000), przez co wynik przestawał się zmieniać przy
    dalszym wzroście temperatury - dokładnie ten objaw zgłosił
    użytkownik. Naprawa: pełne równanie Arrheniusa
    `aging(T) = exp[B * (1/T_rated - 1/T)]`, T w Kelwinach, stała B
    wyprowadzona (nie zgadnięta) z tego samego punktu kalibracji co
    poprzednio (`thermal_halving_deltaT_c` - "podwojenie żywotności przy
    +ΔT" pozostaje dokładnie tym samym punktem odniesienia), ale
    ekstrapolacja poza ten punkt jest teraz fizycznie poprawna i NIE
    potrzebuje niskiego sufitu - prawdziwym, fizycznym ograniczeniem jest
    teraz `decomposition_temp_c` w estimate_remaining_life() niżej, nie
    dowolnie wybrana liczba. Regresja:
    `test_aging_factor_arrhenius_asymetryczny_wzgledem_montsingera` w
    test_cable_life.py."""
    conductor_temp_c = np.asarray(conductor_temp_c, dtype=float)
    t_rated_k = spec.rated_conductor_temp_c + _CELSIUS_TO_KELVIN_OFFSET
    t_half_k = t_rated_k + spec.thermal_halving_deltaT_c
    # B (Kelwiny) wyprowadzone z warunku: aging=2.0 dokladnie przy
    # T = T_rated + thermal_halving_deltaT_c (ten sam punkt kalibracji,
    # ktory dotad definiowal reguly Montsingera).
    b_kelvin = np.log(2.0) * t_rated_k * t_half_k / spec.thermal_halving_deltaT_c
    t_actual_k = conductor_temp_c + _CELSIUS_TO_KELVIN_OFFSET
    aging = np.exp(b_kelvin * (1.0 / t_rated_k - 1.0 / t_actual_k))
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

    # Czy KTÓRAKOLWIEK próbka przekroczyła fizyczną temperaturę rozkładu
    # izolacji (nie średnia - pojedynczy, krótki skok wystarczy, żeby
    # trwale uszkodzić izolację; ta sama logika co graniczne temperatury
    # zwarciowe w IEC 60949, których nie wolno przekroczyć NAWET
    # CHWILOWO). To jest FIZYCZNY sufit modelu - w przeciwieństwie do
    # MAX_AGING_FACTOR (czysto numerycznego zabezpieczenia), tutaj wynik
    # SŁUSZNIE przestaje się różnicować przy dalszym wzroście temperatury:
    # materiał zniszczony to materiał zniszczony, niezależnie o ile
    # stopni przekroczono próg.
    frac_over_decomposition = float(np.mean(temp >= spec.decomposition_temp_c))
    destroyed = frac_over_decomposition > 0.0

    window_years = (len(load_w) / sample_rate_hz) / (365.25 * 24 * 3600)
    equivalent_years_consumed_in_window = window_years * mean_aging

    remaining_design_years = max(0.0, spec.design_life_years - spec.years_in_service)
    cap = spec.design_life_years * _MAX_SENSIBLE_PROJECTION_YEARS_FACTOR
    if destroyed:
        projected_remaining_years = 0.0
    elif mean_aging > 1e-9:
        projected_remaining_years = min(remaining_design_years / mean_aging, cap)
    else:
        projected_remaining_years = cap

    if destroyed:
        status = "ZNISZCZENIE_IZOLACJI"
    elif mean_aging >= 4.0:
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
        "decomposition_temp_c": spec.decomposition_temp_c,
        "frac_samples_over_decomposition_temp": round(frac_over_decomposition, 4),
    }
