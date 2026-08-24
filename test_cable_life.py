"""test_cable_life.py — testy dla cable_life.py (szacowanie żywotności
kabla na podstawie profilu obciążenia)."""

import numpy as np
import pytest

from cable_life import CableSpec, estimate_conductor_temp, estimate_aging_factor, estimate_remaining_life, MAX_AGING_FACTOR


# ---------------------------------------------------------------------
# CableSpec - walidacja i domyślne wartości
# ---------------------------------------------------------------------

def test_spec_domyslne_wartosci_z_insulation_type():
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe")
    assert spec.rated_conductor_temp_c == 90.0
    assert spec.design_life_years == 30.0


def test_spec_pvc_ma_nizsza_temperature_znamionowa():
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="pvc")
    assert spec.rated_conductor_temp_c == 70.0


def test_spec_akceptuje_aluminium():
    spec = CableSpec(rated_load_w=10_000.0, conductor_material="aluminum")
    assert spec.conductor_material == "aluminum"


def test_spec_odrzuca_nieznany_material():
    with pytest.raises(ValueError, match="conductor_material"):
        CableSpec(rated_load_w=10_000.0, conductor_material="unobtainium")


def test_spec_odrzuca_nieznana_izolacje():
    with pytest.raises(ValueError, match="insulation_type"):
        CableSpec(rated_load_w=10_000.0, insulation_type="gumka_do_zucia")


def test_spec_odrzuca_rated_load_ujemne():
    with pytest.raises(ValueError, match="rated_load_w"):
        CableSpec(rated_load_w=-100.0)


def test_spec_odrzuca_ambient_wyzszy_niz_rated():
    with pytest.raises(ValueError, match="rated_conductor_temp_c"):
        CableSpec(rated_load_w=10_000.0, ambient_temp_c=95.0, insulation_type="xlpe")


def test_spec_odrzuca_ujemny_staz():
    with pytest.raises(ValueError, match="years_in_service"):
        CableSpec(rated_load_w=10_000.0, years_in_service=-1.0)


# ---------------------------------------------------------------------
# estimate_conductor_temp
# ---------------------------------------------------------------------

def test_temp_przy_obciazeniu_znamionowym_rowna_sie_temp_znamionowej():
    spec = CableSpec(rated_load_w=10_000.0, ambient_temp_c=25.0, insulation_type="xlpe")
    temp = estimate_conductor_temp(np.array([10_000.0]), spec)
    assert temp[0] == pytest.approx(90.0)


def test_temp_przy_zerowym_obciazeniu_rowna_sie_otoczeniu():
    spec = CableSpec(rated_load_w=10_000.0, ambient_temp_c=25.0)
    temp = estimate_conductor_temp(np.array([0.0]), spec)
    assert temp[0] == pytest.approx(25.0)


def test_temp_polowa_obciazenia_znamionowego_daje_cwiartke_przyrostu():
    """I²R: 50% obciążenia -> 25% przyrostu temperatury (kwadrat)."""
    spec = CableSpec(rated_load_w=10_000.0, ambient_temp_c=25.0, insulation_type="xlpe")  # rise=65
    temp = estimate_conductor_temp(np.array([5_000.0]), spec)
    expected = 25.0 + 65.0 * 0.25
    assert temp[0] == pytest.approx(expected)


def test_temp_rosnie_z_ambient_przy_dowolnym_obciazeniu():
    """Regresja na naprawiony błąd: pierwsza wersja liczyła
    rated_rise = rated_conductor_temp_c - ambient_temp_c, czyli WZGLĘDEM
    TEGO SAMEGO ambient_temp_c, które zaraz potem dodawała z powrotem.
    Przy obciążeniu znamionowym (ratio=1) dawało to wynik CAŁKOWICIE
    NIEZALEŻNY od ambient_temp_c (zawsze = rated_conductor_temp_c), a przy
    przeciążeniu podniesienie ambient wręcz OBNIŻAŁO wynikową temperaturę
    - fizycznie odwrotny kierunek. Teraz: podniesienie ambient o X stopni
    musi podnieść wynikową temperaturę przewodnika o dokładnie X stopni,
    niezależnie od poziomu obciążenia (stałe źródło ciepła I²R nad
    zmiennym otoczeniem)."""
    for load_w in (2_000.0, 10_000.0, 15_000.0):  # niedociążenie / znamionowe / przeciążenie
        temps = []
        for amb in (10.0, 25.0, 40.0, 60.0):
            spec = CableSpec(rated_load_w=10_000.0, ambient_temp_c=amb, insulation_type="xlpe")
            temps.append(estimate_conductor_temp(np.array([load_w]), spec)[0])
        # monotonicznie rosnące wraz z ambient
        assert all(b > a for a, b in zip(temps, temps[1:]))
        # dokladnie 1:1 (staly rated_rise, wiec pochodna po ambient = 1)
        diffs = np.diff(temps)
        assert diffs == pytest.approx(np.diff([10.0, 25.0, 40.0, 60.0]))


# ---------------------------------------------------------------------
# estimate_aging_factor
# ---------------------------------------------------------------------

def test_aging_factor_1_w_temperaturze_znamionowej():
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe")  # rated=90
    aging = estimate_aging_factor(np.array([90.0]), spec)
    assert aging[0] == pytest.approx(1.0)


def test_aging_factor_podwojna_na_deltaT_powyzej():
    """Punkt kalibracji: rownanie Arrheniusa jest wyprowadzone tak, zeby
    DOKLADNIE w tym punkcie (rated + thermal_halving_deltaT_c) dawac
    aging=2.0 - ten sam punkt odniesienia co dawna regula Montsingera."""
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", thermal_halving_deltaT_c=10.0)
    aging = estimate_aging_factor(np.array([100.0]), spec)  # 90+10
    assert aging[0] == pytest.approx(2.0)


def test_aging_factor_arrhenius_asymetryczny_wzgledem_montsingera():
    """Regresja na naprawiony błąd: pierwsza wersja (regula Montsingera,
    2^(deltaT_celsjusz/10)) dawala SYMETRYCZNE podwojenie/polowienie
    zywotnosci dla +deltaT/-deltaT (dokladnie 2.0 i dokladnie 0.5).
    Prawdziwe rownanie Arrheniusa (liniowe w 1/T_bezwzgledne, nie w
    ΔT_celsjusz) jest ASYMETRYCZNE: schlodzenie o ΔT daje WIEKSZY zysk
    zywotnosci niz podgrzanie o ΔT ja kosztuje - wiec aging przy rated-10
    NIE jest dokladnie 0.5, tylko nieco mniej (silniejszy zysk z
    chlodzenia), mimo ze aging przy rated+10 pozostaje dokladnie 2.0 (tak
    zdefiniowany punkt kalibracji)."""
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", thermal_halving_deltaT_c=10.0)
    aging_below = estimate_aging_factor(np.array([80.0]), spec)[0]  # 90-10
    aging_above = estimate_aging_factor(np.array([100.0]), spec)[0]  # 90+10
    assert aging_above == pytest.approx(2.0)
    assert aging_below < 0.5  # asymetria: wiekszy zysk z chlodzenia
    assert aging_below == pytest.approx(0.4807526920460988, rel=1e-6)
    # sanity: odwrotnosc nie jest dokladnie symetryczna (1/2.0=0.5 != aging_below)
    assert aging_below != pytest.approx(1.0 / aging_above)


def test_aging_factor_ograniczony_przy_ekstremalnej_temperaturze():
    """MAX_AGING_FACTOR to teraz czysto numeryczny sufit bezpieczenstwa
    (przed przepelnieniem/nieczytelnymi liczbami), nie glowny mechanizm
    ochronny - ten pelni teraz decomposition_temp_c (patrz
    test_remaining_life_destrukcja_izolacji_*). Test: przy ASTRONOMICZNIE
    duzej delcie (+1000°C) i tak trzeba gdzies uciac liczbe."""
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", thermal_halving_deltaT_c=10.0)
    aging = estimate_aging_factor(np.array([90.0 + 1000.0]), spec)
    assert aging[0] == pytest.approx(MAX_AGING_FACTOR)


# ---------------------------------------------------------------------
# estimate_remaining_life
# ---------------------------------------------------------------------

def test_remaining_life_pusta_seria_rzuca_blad():
    spec = CableSpec(rated_load_w=10_000.0)
    with pytest.raises(ValueError, match="pusta"):
        estimate_remaining_life(np.array([]), sample_rate_hz=100.0, spec=spec)


def test_remaining_life_niepoprawny_sample_rate_rzuca_blad():
    spec = CableSpec(rated_load_w=10_000.0)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        estimate_remaining_life(np.array([1000.0]), sample_rate_hz=0.0, spec=spec)


def test_remaining_life_lekkie_obciazenie_status_ok_i_dluzsza_prognoza():
    """Kabel pracujący daleko poniżej znamionowego obciążenia powinien
    mieć aging_factor < 1 i status OK - żywotność 'zużywana' wolniej
    niż projektowo."""
    spec = CableSpec(rated_load_w=10_000.0, design_life_years=30.0, years_in_service=5.0)
    load = np.full(1000, 2000.0)  # 20% obciążenia znamionowego
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["status"] == "OK"
    assert result["mean_aging_factor"] < 1.0
    assert result["estimated_remaining_years"] > (30.0 - 5.0)


def test_remaining_life_chroniczne_przeciazenie_daje_krytyczny_status():
    spec = CableSpec(rated_load_w=10_000.0, design_life_years=30.0, thermal_halving_deltaT_c=10.0)
    load = np.full(1000, 20_000.0)  # 200% obciążenia znamionowego -> bardzo gorąco
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["status"] == "KRYTYCZNE"
    assert result["mean_aging_factor"] > 4.0
    assert result["estimated_remaining_years"] < (30.0 / 4.0)


def test_remaining_life_projekcja_ograniczona_gorna_granica():
    """Przy obciążeniu ~0 aging_factor -> 0, projekcja nie powinna
    eksplodować do absurdalnych tysięcy lat - jest ograniczona."""
    spec = CableSpec(rated_load_w=10_000.0, design_life_years=30.0)
    load = np.full(1000, 1.0)  # praktycznie zero obciążenia
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["estimated_remaining_years"] <= 30.0 * 3.0


def test_remaining_life_praca_dokladnie_znamionowa_konsumuje_projektowo():
    spec = CableSpec(rated_load_w=10_000.0, design_life_years=30.0, years_in_service=0.0)
    load = np.full(1000, 10_000.0)  # dokładnie 100% znamionowego
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["mean_aging_factor"] == pytest.approx(1.0, abs=1e-6)
    assert result["estimated_remaining_years"] == pytest.approx(30.0, abs=0.1)


def test_remaining_life_niedopasowany_rated_load_nie_daje_absurdalnego_wspolczynnika():
    """Regresja na realny przypadek z dashboardu: rated_load dobrane
    znacznie niżej niż faktyczny profil obciążenia (np. użytkownik
    wpisał małą moc znamionową do scenariusza demo, który generuje
    znacznie wyższe waty) - kilka próbek z ratio~2.6x potrafiło wcześniej
    wywindować mean_aging_factor do dziesiątek tysięcy.

    Z fizycznym sufitem (decomposition_temp_c) status jest teraz JESZCZE
    trafniejszy niż wcześniej: regularne skoki do ~2.6x znamionowej na
    izolacji PVC (rated=70°C, rozkład ~160°C) dają chwilowe temperatury
    grubo powyżej progu rozkładu - w rzeczywistości takie powtarzające
    się skoki NAPRAWDĘ zniszczyłyby izolację PVC, więc status
    ZNISZCZENIE_IZOLACJI (nie tylko "KRYTYCZNE") jest poprawną,
    ostrzejszą odpowiedzią, nie regresją."""
    rng = np.random.default_rng(3)
    n = 2000
    # profil głównie umiarkowany, ale z okresowymi skokami do ~2.6x znamionowej
    load = np.full(n, 3000.0) + rng.normal(0, 100, n)
    load[::50] = 11_766.0  # regularne skoki przeciążenia (co 50-tą próbkę)
    spec = CableSpec(rated_load_w=4_500.0, insulation_type="pvc", years_in_service=10.0)
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["mean_aging_factor"] <= MAX_AGING_FACTOR
    assert result["status"] == "ZNISZCZENIE_IZOLACJI"
    assert result["estimated_remaining_years"] == 0.0
    assert result["frac_samples_over_decomposition_temp"] > 0.0


# ---------------------------------------------------------------------
# decomposition_temp_c: fizyczny (nie numeryczny) sufit modelu
# ---------------------------------------------------------------------

def test_remaining_life_destrukcja_izolacji_gdy_chocby_jedna_probka_przekracza_rozklad():
    """Nawet POJEDYNCZY, krotki skok powyzej temperatury rozkladu
    izolacji wystarczy, zeby oznaczyc status jako zniszczenie - nie trzeba
    sredniej ponad prog (analogia: graniczne temperatury zwarciowe wg
    IEC 60949, ktorych nie wolno przekroczyc nawet chwilowo)."""
    n = 500
    load = np.full(n, 1000.0)  # spokojne, dalekie od znamionowego
    load[250] = 50_000.0  # JEDEN skrajny, chwilowy skok
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", years_in_service=0.0)
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["status"] == "ZNISZCZENIE_IZOLACJI"
    assert result["estimated_remaining_years"] == 0.0
    assert result["frac_samples_over_decomposition_temp"] == pytest.approx(1.0 / n)


def test_remaining_life_bez_przekroczenia_rozkladu_normalny_status():
    """Kontrola: profil, ktory NIE przekracza temperatury rozkladu, nie
    powinien byc oznaczony jako zniszczenie, nawet przy podwyzszonym
    ambient."""
    n = 500
    load = np.full(n, 10_000.0)  # dokladnie znamionowe
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", ambient_temp_c=40.0)
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["status"] != "ZNISZCZENIE_IZOLACJI"
    assert result["frac_samples_over_decomposition_temp"] == 0.0


def test_spec_odrzuca_decomposition_ponizej_rated():
    with pytest.raises(ValueError, match="decomposition_temp_c"):
        CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", decomposition_temp_c=50.0)
