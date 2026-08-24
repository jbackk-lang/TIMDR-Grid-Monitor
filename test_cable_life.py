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


# ---------------------------------------------------------------------
# estimate_aging_factor
# ---------------------------------------------------------------------

def test_aging_factor_1_w_temperaturze_znamionowej():
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe")  # rated=90
    aging = estimate_aging_factor(np.array([90.0]), spec)
    assert aging[0] == pytest.approx(1.0)


def test_aging_factor_polowa_na_deltaT_ponizej():
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", thermal_halving_deltaT_c=10.0)
    aging = estimate_aging_factor(np.array([80.0]), spec)  # 90-10
    assert aging[0] == pytest.approx(0.5)


def test_aging_factor_podwojna_na_deltaT_powyzej():
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", thermal_halving_deltaT_c=10.0)
    aging = estimate_aging_factor(np.array([100.0]), spec)  # 90+10
    assert aging[0] == pytest.approx(2.0)


def test_aging_factor_ograniczony_przy_ekstremalnej_temperaturze():
    """Regresja: bez capu, 2^(delta/10) przy dużej delcie (np. skrajne
    przeciążenie z niedopasowanym rated_load) daje wartości rzędu
    dziesiątek/setek tysięcy - nieczytelne i bez dodatkowej informacji
    (kabel dawno przekroczyłby dopuszczalną temperaturę w praktyce)."""
    spec = CableSpec(rated_load_w=10_000.0, insulation_type="xlpe", thermal_halving_deltaT_c=10.0)
    aging = estimate_aging_factor(np.array([90.0 + 300.0]), spec)  # +300°C ponad znamionową
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
    wywindować mean_aging_factor do dziesiątek tysięcy. Teraz ograniczone."""
    rng = np.random.default_rng(3)
    n = 2000
    # profil głównie umiarkowany, ale z okresowymi skokami do ~2.6x znamionowej
    load = np.full(n, 3000.0) + rng.normal(0, 100, n)
    load[::50] = 11_766.0  # regularne skoki przeciążenia (co 50-tą próbkę)
    spec = CableSpec(rated_load_w=4_500.0, insulation_type="pvc", years_in_service=10.0)
    result = estimate_remaining_life(load, sample_rate_hz=100.0, spec=spec)
    assert result["mean_aging_factor"] <= MAX_AGING_FACTOR
    assert result["status"] == "KRYTYCZNE"  # to i tak realne, chroniczne przeciążenie
