"""test_csv_loader.py — testy dla csv_loader.py (import CSV/Excel)."""

import numpy as np
import pandas as pd
import pytest

from csv_loader import load_csv, CsvLoaderError


def _write_csv(tmp_path, df, name="grid.csv"):
    path = str(tmp_path / name)
    df.to_csv(path, index=False)
    return path


def _sample_df(n=200, freq_seconds=1.0):
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-01-01", periods=n, freq=pd.Timedelta(seconds=freq_seconds))
    return pd.DataFrame({
        "timestamp": ts,
        "voltage": 230.0 + rng.normal(0, 0.5, n),
        "frequency": 50.0 + rng.normal(0, 0.02, n),
        "THD_U": np.abs(rng.normal(2.0, 0.3, n)),
        "load": 2000.0 + rng.normal(0, 50, n),
    })


def test_load_csv_rozpoznaje_standardowe_naglowki(tmp_path):
    path = _write_csv(tmp_path, _sample_df())
    signals, rate = load_csv(path)
    assert len(signals.voltage) == 200
    assert abs(rate - 1.0) < 1e-6


def test_load_csv_rozpoznaje_alternatywne_naglowki_polskie(tmp_path):
    df = _sample_df().rename(columns={
        "voltage": "Napięcie", "frequency": "Częstotliwość",
        "THD_U": "harmoniczne", "load": "Obciążenie", "timestamp": "Data_Czas",
    })
    path = _write_csv(tmp_path, df)
    signals, rate = load_csv(path)
    assert len(signals.load) == 200


def test_load_csv_jawne_mapowanie_kolumn(tmp_path):
    df = _sample_df().rename(columns={"voltage": "U1", "load": "MOC_CALK"})
    path = _write_csv(tmp_path, df)
    signals, rate = load_csv(path, column_map={"voltage": "U1", "load": "MOC_CALK"})
    assert len(signals.voltage) == 200


def test_load_csv_nieznana_kolumna_daje_czytelny_blad(tmp_path):
    df = _sample_df().drop(columns=["voltage"])
    path = _write_csv(tmp_path, df)
    with pytest.raises(CsvLoaderError, match="voltage"):
        load_csv(path)


def test_load_csv_plik_nie_istnieje():
    with pytest.raises(CsvLoaderError, match="nie istnieje"):
        load_csv("/tmp/nieistniejacy_plik_xyz_12345.csv")


def test_load_csv_pusty_plik_daje_czytelny_blad(tmp_path):
    path = str(tmp_path / "empty.csv")
    pd.DataFrame(columns=["timestamp", "voltage", "frequency", "THD_U", "load"]).to_csv(path, index=False)
    with pytest.raises(CsvLoaderError, match="nie zawiera"):
        load_csv(path)


def test_load_csv_nienumeryczne_wartosci_dają_czytelny_blad(tmp_path):
    df = _sample_df()
    df.loc[5, "voltage"] = "USZKODZONE"
    path = _write_csv(tmp_path, df)
    with pytest.raises(CsvLoaderError, match="nienumerycznych"):
        load_csv(path)


def test_load_csv_brakujace_wartosci_daja_czytelny_blad(tmp_path):
    df = _sample_df()
    df.loc[5, "load"] = np.nan
    path = _write_csv(tmp_path, df)
    with pytest.raises(CsvLoaderError, match="brakujące"):
        load_csv(path)


def test_load_csv_konwersja_kw_na_w(tmp_path):
    df = _sample_df()
    df["load"] = df["load"] / 1000.0  # teraz w kW
    path = _write_csv(tmp_path, df)
    signals, rate = load_csv(path, load_unit="kW")
    assert signals.load.mean() > 1000  # z powrotem w W rzędu tysięcy


def test_load_csv_niepoprawna_jednostka_rzuca_blad(tmp_path):
    path = _write_csv(tmp_path, _sample_df())
    with pytest.raises(CsvLoaderError, match="jednostka"):
        load_csv(path, load_unit="MW")


def test_load_csv_probkowanie_wyliczone_z_timestampow(tmp_path):
    path = _write_csv(tmp_path, _sample_df(n=100, freq_seconds=0.1))
    signals, rate = load_csv(path)
    assert abs(rate - 10.0) < 0.5  # 0.1s odstęp = 10Hz


def test_load_csv_bez_kolumny_czasu_wymaga_jawnego_sample_rate(tmp_path):
    df = _sample_df().drop(columns=["timestamp"])
    path = _write_csv(tmp_path, df)
    with pytest.raises(CsvLoaderError, match="sample_rate_hz"):
        load_csv(path)
    # z jawnym sample_rate_hz dziala poprawnie
    signals, rate = load_csv(path, sample_rate_hz=5.0)
    assert rate == 5.0


def test_load_csv_niewspierane_rozszerzenie(tmp_path):
    path = str(tmp_path / "grid.txt")
    _sample_df().to_csv(path, index=False)
    with pytest.raises(CsvLoaderError, match="rozszerzenie"):
        load_csv(path)


def test_load_excel(tmp_path):
    pytest.importorskip("openpyxl")
    path = str(tmp_path / "grid.xlsx")
    _sample_df().to_excel(path, index=False)
    signals, rate = load_csv(path)
    assert len(signals.voltage) == 200
