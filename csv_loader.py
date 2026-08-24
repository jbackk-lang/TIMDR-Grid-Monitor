"""
csv_loader.py — import danych sieci energetycznej z pliku CSV/Excel
================================================================================
Wczytywanie logów z liczników/analizatorów jakości energii (eksporty typu
Fluke/Janitza/inne). Jawna walidacja schematu - NIGDY bare `except: pass`
(lekcja ze Synoptyka: schema mismatch to pułapka cichego zawodzenia -
błędny/nierozpoznany format pliku musi dać czytelny błąd, nie cichą,
pustą/błędną analizę).

Rozpoznawane aliasy nazw kolumn (bez rozróżniania wielkości liter,
spacje/podkreślniki równoważne) - jeśli plik ma inne nazwy, podaj jawne
mapowanie przez `column_map`.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from grid_monitor import TimdrEnergySignals


class CsvLoaderError(Exception):
    """Jawny błąd wczytywania/parsowania pliku - nigdy cichy pusty wynik."""


COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "data_czas", "datetime", "czas", "data"],
    "voltage": ["voltage", "u", "u_l1", "napiecie", "napięcie", "v", "volt"],
    "frequency": ["frequency", "f", "czestotliwosc", "częstotliwość", "hz", "freq"],
    "harmonics": ["thd", "thd_u", "harmonics", "harmoniczne", "zniek_harm"],
    "load": ["load", "p", "power", "obciazenie", "obciążenie", "moc", "watts", "w"],
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[\s_]+", "_", str(name).strip().lower())


def _find_column(columns_normalized: dict, canonical: str, column_map: dict | None) -> str:
    if column_map and canonical in column_map:
        wanted = _normalize_name(column_map[canonical])
        if wanted in columns_normalized:
            return columns_normalized[wanted]
        raise CsvLoaderError(
            f"Podano w column_map['{canonical}']='{column_map[canonical]}', "
            f"ale takiej kolumny nie ma w pliku. Dostępne kolumny: "
            f"{list(columns_normalized.values())}"
        )
    for alias in COLUMN_ALIASES[canonical]:
        if alias in columns_normalized:
            return columns_normalized[alias]
    raise CsvLoaderError(
        f"Nie znaleziono kolumny dla '{canonical}' (szukane aliasy: "
        f"{COLUMN_ALIASES[canonical]}). Dostępne kolumny w pliku: "
        f"{list(columns_normalized.values())}. Podaj jawne mapowanie przez "
        f"column_map={{'{canonical}': 'nazwa_twojej_kolumny'}}."
    )


def load_csv(
    path: str,
    column_map: dict | None = None,
    load_unit: str = "W",
    sample_rate_hz: float | None = None,
) -> tuple[TimdrEnergySignals, float]:
    """
    Wczytuje CSV lub Excel (.xlsx/.xls, rozpoznawane po rozszerzeniu) z
    logiem pomiarów sieci energetycznej. Zwraca (signals, sample_rate_hz).

    `load_unit`: "W" albo "kW" - jeśli plik ma obciążenie w kW, poda się
    "kW" i wartości zostaną przeliczone na W (spójne z resztą repo).

    `sample_rate_hz`: jeśli None, wyliczana automatycznie z mediany
    odstępów czasowych w kolumnie timestamp (wymaga rozpoznawalnej
    kolumny czasu w formacie parsowalnym przez pandas).
    """
    if not os.path.exists(path):
        raise CsvLoaderError(f"Plik nie istnieje: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise CsvLoaderError(
                f"Nieobsługiwane rozszerzenie pliku '{ext}' - obsługiwane: .csv, .xlsx, .xls"
            )
    except CsvLoaderError:
        raise
    except Exception as e:
        # Jawnie opakowany błąd parsowania - nigdy cichy pusty DataFrame.
        raise CsvLoaderError(f"Błąd odczytu/parsowania pliku '{path}': {e}") from e

    if df.empty:
        raise CsvLoaderError(f"Plik '{path}' nie zawiera żadnych wierszy danych.")

    columns_normalized = {_normalize_name(c): c for c in df.columns}

    voltage_col = _find_column(columns_normalized, "voltage", column_map)
    frequency_col = _find_column(columns_normalized, "frequency", column_map)
    harmonics_col = _find_column(columns_normalized, "harmonics", column_map)
    load_col = _find_column(columns_normalized, "load", column_map)

    for col_name, series_name in [
        (voltage_col, "napięcie"), (frequency_col, "częstotliwość"),
        (harmonics_col, "THD"), (load_col, "obciążenie"),
    ]:
        if not pd.api.types.is_numeric_dtype(df[col_name]):
            coerced = pd.to_numeric(df[col_name], errors="coerce")
            n_bad = coerced.isna().sum() - df[col_name].isna().sum()
            if n_bad > 0:
                raise CsvLoaderError(
                    f"Kolumna '{col_name}' ({series_name}) zawiera {n_bad} "
                    f"nienumerycznych wartości, których nie da się zinterpretować "
                    f"jako liczby - sprawdź format pliku."
                )
            df[col_name] = coerced

    if df[[voltage_col, frequency_col, harmonics_col, load_col]].isna().any().any():
        raise CsvLoaderError(
            "Plik zawiera brakujące (puste/NaN) wartości w co najmniej jednej z "
            "wymaganych kolumn (napięcie/częstotliwość/THD/obciążenie) - uzupełnij "
            "lub usuń niekompletne wiersze przed importem."
        )

    load_values = df[load_col].to_numpy(dtype=float)
    if load_unit.upper() == "KW":
        load_values = load_values * 1000.0
    elif load_unit.upper() != "W":
        raise CsvLoaderError(f"Nieobsługiwana jednostka load_unit='{load_unit}' - użyj 'W' albo 'kW'.")

    signals = TimdrEnergySignals(
        voltage=df[voltage_col].to_numpy(dtype=float),
        frequency=df[frequency_col].to_numpy(dtype=float),
        harmonics=df[harmonics_col].to_numpy(dtype=float),
        load=load_values,
    )

    if sample_rate_hz is not None:
        resolved_rate = sample_rate_hz
    else:
        ts_key = "timestamp"
        if column_map and ts_key in column_map:
            ts_col = columns_normalized.get(_normalize_name(column_map[ts_key]))
        else:
            ts_col = next(
                (columns_normalized[a] for a in COLUMN_ALIASES["timestamp"] if a in columns_normalized),
                None,
            )
        if ts_col is None:
            raise CsvLoaderError(
                "Nie podano sample_rate_hz i nie znaleziono kolumny czasu do jego "
                "wyliczenia - podaj sample_rate_hz jawnie albo dodaj rozpoznawalną "
                "kolumnę czasu (np. 'timestamp')."
            )
        try:
            ts = pd.to_datetime(df[ts_col])
        except Exception as e:
            raise CsvLoaderError(
                f"Nie udało się rozpoznać kolumny czasu '{ts_col}' jako dat/czasów: {e}"
            ) from e
        deltas = ts.diff().dropna().dt.total_seconds()
        deltas = deltas[deltas > 0]
        if len(deltas) == 0:
            raise CsvLoaderError(
                "Nie udało się wyliczyć częstotliwości próbkowania z kolumny czasu "
                "(brak poprawnych, rosnących odstępów) - podaj sample_rate_hz jawnie."
            )
        resolved_rate = 1.0 / float(np.median(deltas))

    return signals, resolved_rate
