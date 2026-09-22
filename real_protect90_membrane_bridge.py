"""real_protect90_membrane_bridge.py

protect90_membrane_bridge -- widmo macierzy korelacji miedzy
jednoczesnymi fazami pradu (i napiecia) sieci PROTECT-90, na oknach
baseline (przed t_evnt_start) vs event (po t_evnt_start), na 512
zamrozonych epizodach zwarciowych.

Kod matematyczny (macierz korelacji, widmo, spectral_concentration,
participation_ratio, membrane_window_metrics) jest w calosci REUZYTY,
BEZ ZMIAN, z `timdr_geometry/spectral_family.py` (patrz
docs/PREREG_PROTECT90_MEMBRANE_v0.1.md w tym repo po pelny kontekst i
uzasadnienie kazdej decyzji). Ten plik dodaje WYLACZNIE: (1) wczytanie
danych PROTECT-90 i wyciecie okien baseline/event zgodnie z
`analysis_rule` z manifestu zbioru, (2) test Wilcoxona sparowanego
(baseline vs event NA TYM SAMYM epizodzie -- naturalne parowanie, w
odroznieniu od testu Manna-Whitneya uzywanego w domenie lozyskowej,
gdzie normal/fault byly niezaleznymi grupami plikow), (3) checkpointy
JSON, zeby przetrwac limit czasu pojedynczego wywolania w tym
sandboksie.

Lokalizacja: `Bus_1_Line_01_02A` (pierwsza alfabetycznie sposrod 8
dostepnych -- regula ustalona PRZED zobaczeniem wynikow, PREREG SS2.1).
Test PRIMARNY: N=3 (cur_L1,cur_L2,cur_L3). Test SEKUNDARNY: N=6
(cur_L1..L3, vol_L1..L3).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# sys.path -- reuzycie spectral_family.py ORAZ timdr_formalism.pipeline
# bez kopiowania kodu. Oba repo (TIMDR-Geometry-Formalism,
# TIMDR-Math-Formalism) istnieja w dwoch lokalizacjach -- probuj
# najpierw samodzielne repo siostrzane wobec TIMDR-Grid-Monitor, potem
# kopie zagniezdzona w GIA-TIMDR (ten sam wzorzec co ponizej dla
# spectral_family, patrz `timdr_time/_bootstrap.py` w
# TIMDR-Time-Formalism po analogiczny wzorzec "sibling-repos").
# ---------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
_CANDIDATES = [
    os.path.join(_PARENT, "TIMDR-Geometry-Formalism", "timdr_geometry"),
    os.path.join(_PARENT, "GIA-TIMDR", "TIMDR-Geometry-Formalism", "timdr_geometry"),
]
for _c in _CANDIDATES:
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)
        break
else:
    raise ImportError(
        "Nie znaleziono spectral_family.py w zadnej ze znanych lokalizacji: "
        + ", ".join(_CANDIDATES)
    )

import spectral_family as sf  # noqa: E402

# timdr_formalism JEST prawdziwym pakietem Pythona (ma __init__.py), wiec
# na sys.path trzeba wstawic KORZEN repo (katalog NADRZEDNY wobec
# `timdr_formalism/`), nie sam katalog pakietu -- w odroznieniu od
# spectral_family.py wyzej, ktory jest pojedynczym plikiem importowanym
# bezposrednio (plaski import, bez pakietu).
_TF_CANDIDATES = [
    os.path.join(_PARENT, "TIMDR-Math-Formalism"),
    os.path.join(_PARENT, "GIA-TIMDR", "TIMDR-Math-Formalism"),
]
for _tf_c in _TF_CANDIDATES:
    if os.path.isdir(_tf_c) and _tf_c not in sys.path:
        sys.path.insert(0, _tf_c)
        break
else:
    raise ImportError(
        "Nie znaleziono pakietu timdr_formalism w zadnej ze znanych "
        "lokalizacji: " + ", ".join(_TF_CANDIDATES)
    )

# Test Wilcoxona sparowany -- kanoniczna implementacja (ze znaleziona i
# naprawiona w TEJ sesji poprawka bledu znaku scipy) zyje teraz w
# timdr_formalism/pipeline.py::paired_wilcoxon_test, WSPOLNA dla
# wszystkich mostow GIA-TIMDR, nie tylko PROTECT-90 -- patrz obszerna
# uwaga o `z=-abs(z)` w scipy nad ta funkcja w pipeline.py.
from timdr_formalism.pipeline import (  # noqa: E402
    paired_wilcoxon_test as _tf_paired_wilcoxon_test,
)

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_stats = None
    _HAS_SCIPY = False

# ---------------------------------------------------------------------
# Stale zamrozone w PREREG_PROTECT90_MEMBRANE_v0.1.md
# ---------------------------------------------------------------------

DATA_ROOT = Path(_THIS_DIR) / "data" / "protect90"
FROZEN_DIR = DATA_ROOT / "frozen"
EPISODE_IDS_FILE = FROZEN_DIR / "B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json"
LABELS_FILE = DATA_ROOT / "raw" / "hv_double_line_90kv_labels.csv"
PKL_DIR = FROZEN_DIR / "preprocessed_data"
SAMPLE_RATE_HZ = 6400

LOCATION = "Bus_1_Line_01_02A"
CHANNELS_N3 = [f"{LOCATION}_cur_L{k}_A" for k in (1, 2, 3)]
CHANNELS_N6 = CHANNELS_N3 + [f"{LOCATION}_vol_L{k}_V" for k in (1, 2, 3)]

WINDOW_SIZES = (128, 256)
ALPHA = 0.05
CHECKPOINT_EVERY = 64

RAW_RESULTS_PATH = Path(_THIS_DIR) / "protect90_membrane_raw_results.json"


# ---------------------------------------------------------------------
# Test Wilcoxona sparowany -- CIENKI WRAPPER na kanoniczna implementacje
# w `timdr_formalism.pipeline.paired_wilcoxon_test` (wydzielona tam z
# TEGO pliku podczas tej sesji, PO tym jak zostal tu znaleziony i
# naprawiony blad znaku `scipy.stats.wilcoxon` -- patrz obszerna uwaga
# nad `paired_wilcoxon_test` w pipeline.py po pelny opis bledu
# `z=-abs(z)` scipy dla testu two-sided oraz uzasadnienie, dlaczego znak
# liczony jest wlasna implementacja, a scipy sluzy WYLACZNIE jako
# cross-check p-wartosci). Ten wrapper istnieje TYLKO po to, zeby
# zachowac dokladnie ten sam ksztalt dict (`t["pvalue"]`,
# `t["effect_size_r"]`, `t.get("inconclusive")`, ...), ktorego
# oczekuja `analyze()`/`classify()` ponizej w tym pliku -- logika
# matematyczna w calosci zyje teraz w pipeline.py, NIE tutaj.
# ---------------------------------------------------------------------


def paired_wilcoxon_test(event_values: np.ndarray, baseline_values: np.ndarray) -> Dict:
    """Test Wilcoxona ze znakiem rang (sparowany, dwustronny) miedzy
    `event_values` (test) i `baseline_values` (tlo), sparowane PO
    INDEKSIE (ten sam epizod). Deleguje do
    `timdr_formalism.pipeline.paired_wilcoxon_test` i konwertuje zwrocony
    `TestResult` (dataclass) na dict o DOKLADNIE tych samych kluczach co
    poprzednia (inline) implementacja tego modulu -- p-wartosc,
    statystyka z, rozmiar efektu (rank-biserial dla par `r=z/sqrt(n)`),
    liczba par, liczba par niezerowych (po usunieciu remisow
    baseline==event, konwencja `zero_method="wilcox"`)."""
    result = _tf_paired_wilcoxon_test(event_values, baseline_values)
    out: Dict = {
        "n_pairs": result.n_test,
        "n_nonzero": result.n_nonzero,
        "n_nan_dropped": result.n_nan_dropped,
        "pvalue": result.pvalue,
        "zstatistic": result.statistic,
        "effect_size_r": result.effect_size_r,
        "median_event": result.median_test,
        "median_baseline": result.median_background,
        "inconclusive": result.inconclusive,
    }
    if result.inconclusive:
        # Ta sama konwencja co poprzednia inline implementacja: klucz
        # "reason" obecny WYLACZNIE w galezi inconclusive (nie zawsze),
        # zeby nie zmieniac ksztaltu dict, ktorego moga oczekiwac
        # istniejace testy/raporty.
        out["reason"] = result.reason
    return out


def effect_size_label(r: float) -> str:
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "n/a"
    ar = abs(r)
    if ar < 0.1:
        return "pomijalny"
    if ar < 0.3:
        return "maly"
    if ar < 0.5:
        return "sredni"
    return "duzy"


# ---------------------------------------------------------------------
# Wczytanie danych i wyciecie okien baseline/event
# ---------------------------------------------------------------------


def load_frozen_episode_ids() -> List[int]:
    data = json.loads(EPISODE_IDS_FILE.read_text(encoding="utf-8"))
    return [int(x) for x in data["sample_ids"]]


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_FILE)
    return df.set_index("sample_id")


def episode_path(sample_id: int) -> Path:
    return PKL_DIR / f"{sample_id}_sample_hv_double_line_90kv.pkl"


def event_start_index(time_s: np.ndarray, t_evnt_start: float) -> int:
    """Pierwszy indeks probki z `time_s >= t_evnt_start` (wyszukiwanie
    binarne, nie zaokraglenie recznie)."""
    return int(np.searchsorted(time_s, t_evnt_start, side="left"))


def compute_episode_metrics(sample_id: int, t_evnt_start: float) -> Optional[Dict]:
    """Dla jednego epizodu: wczytaj pkl, wytnij baseline/event dla obu
    rozmiarow okna, policz `spectral_concentration`/`participation_ratio`/
    `membrane_spectral_ratio` (`membrane_window_metrics` z
    `spectral_family.py`, bez zmian) dla N=3 i N=6. Zwraca None (z
    odnotowaniem w wywolujacym), jesli KTORYKOLWIEK rozmiar okna nie
    miesci sie w epizodzie (filtr dlugosci z PREREG SS3.1)."""
    path = episode_path(sample_id)
    if not path.is_file():
        return None
    df = pd.read_pickle(path)
    time_s = df["time_s"].to_numpy()
    n_total = len(time_s)
    idx = event_start_index(time_s, t_evnt_start)

    per_window = {}
    rejected_windows = []
    for w in WINDOW_SIZES:
        if idx < w or idx + w > n_total:
            rejected_windows.append(w)
            continue
        baseline_slice = slice(idx - w, idx)
        event_slice = slice(idx, idx + w)

        for n_label, cols in (("N3", CHANNELS_N3), ("N6", CHANNELS_N6)):
            base_channels = [df[c].to_numpy()[baseline_slice] for c in cols]
            event_channels = [df[c].to_numpy()[event_slice] for c in cols]
            m_base = sf.membrane_window_metrics(base_channels)
            m_event = sf.membrane_window_metrics(event_channels)
            per_window[f"w{w}_{n_label}"] = {
                "baseline": m_base,
                "event": m_event,
            }

    return {
        "sample_id": sample_id,
        "event_start_idx": idx,
        "n_total": n_total,
        "rejected_windows": rejected_windows,
        "metrics": per_window,
    }


# ---------------------------------------------------------------------
# Petla glowna z checkpointem JSON (odpornosc na limit czasu bash)
# ---------------------------------------------------------------------


def run_extraction(max_episodes: Optional[int] = None, time_budget_s: float = 90.0) -> Dict:
    """Wczytuje/tworzy `protect90_membrane_raw_results.json`, przetwarza
    epizody, ktorych jeszcze NIE MA w checkpointcie, zapisuje co
    `CHECKPOINT_EVERY` epizodow ORAZ gdy zblizy sie budzet czasu tego
    wywolania -- bezpieczne do wielokrotnego, przyrostowego
    uruchamiania w tym samym procesie sandboksa."""
    t0 = time.time()
    episode_ids = load_frozen_episode_ids()
    labels = load_labels()

    if RAW_RESULTS_PATH.is_file():
        state = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        state = {"episodes": {}, "missing_file": [], "rejected": {}}

    done_ids = set(int(k) for k in state["episodes"].keys())
    todo = [sid for sid in episode_ids if sid not in done_ids]
    if max_episodes is not None:
        todo = todo[:max_episodes]

    n_processed_this_run = 0
    for sample_id in todo:
        if sample_id not in labels.index:
            state["missing_file"].append(sample_id)
            continue
        t_evnt_start = float(labels.loc[sample_id, "t_evnt_start"])
        result = compute_episode_metrics(sample_id, t_evnt_start)
        if result is None:
            state["missing_file"].append(sample_id)
        else:
            if result["rejected_windows"]:
                state["rejected"][str(sample_id)] = result["rejected_windows"]
            state["episodes"][str(sample_id)] = result

        n_processed_this_run += 1
        if n_processed_this_run % CHECKPOINT_EVERY == 0:
            RAW_RESULTS_PATH.write_text(json.dumps(state), encoding="utf-8")

        if time.time() - t0 > time_budget_s:
            break

    RAW_RESULTS_PATH.write_text(json.dumps(state), encoding="utf-8")
    n_done = len(state["episodes"])
    n_target = len(episode_ids)
    return {
        "n_done": n_done,
        "n_target": n_target,
        "complete": n_done + len(state["missing_file"]) >= n_target,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------
# Analiza: baseline vs event, test Wilcoxona sparowany, stabilnosc znaku
# ---------------------------------------------------------------------


def analyze() -> Dict:
    state = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    episodes = state["episodes"]

    report: Dict = {"n_episodes_total": len(episodes), "windows": {}}

    for w in WINDOW_SIZES:
        report["windows"][str(w)] = {}
        for n_label in ("N3", "N6"):
            key = f"w{w}_{n_label}"
            base_conc, event_conc = [], []
            base_pr, event_pr = [], []
            base_ratio, event_ratio = [], []
            n_missing = 0
            for sid, ep in episodes.items():
                m = ep["metrics"].get(key)
                if m is None:
                    n_missing += 1
                    continue
                base_conc.append(m["baseline"]["spectral_concentration"])
                event_conc.append(m["event"]["spectral_concentration"])
                base_pr.append(m["baseline"]["participation_ratio"])
                event_pr.append(m["event"]["participation_ratio"])
                base_ratio.append(m["baseline"]["membrane_spectral_ratio"])
                event_ratio.append(m["event"]["membrane_spectral_ratio"])

            conc_test = paired_wilcoxon_test(np.array(event_conc), np.array(base_conc))
            pr_test = paired_wilcoxon_test(np.array(event_pr), np.array(base_pr))
            ratio_test = paired_wilcoxon_test(np.array(event_ratio), np.array(base_ratio))

            report["windows"][str(w)][n_label] = {
                "n_missing_this_cell": n_missing,
                "spectral_concentration": conc_test,
                "participation_ratio": pr_test,
                "membrane_spectral_ratio": ratio_test,
            }

    # stabilnosc znaku na spectral_concentration, per N, miedzy oknami
    report["sign_stability"] = {}
    for n_label in ("N3", "N6"):
        signs = []
        for w in WINDOW_SIZES:
            r = report["windows"][str(w)][n_label]["spectral_concentration"]["effect_size_r"]
            signs.append(None if (r is None or math.isnan(r)) else (r > 0))
        stable = len(set(s for s in signs if s is not None)) <= 1 and None not in signs
        report["sign_stability"][n_label] = {
            "signs_by_window": dict(zip([str(w) for w in WINDOW_SIZES], signs)),
            "stable": stable,
        }

    return report


def classify(report: Dict) -> Dict:
    """SUPPORTED / NOT_SUPPORTED / NIESTABILNY / INCONCLUSIVE wg
    PREREG_PROTECT90_MEMBRANE_v0.1.md SS5, dla kazdego N osobno,
    metryka klasyfikujaca = spectral_concentration."""
    out = {}
    for n_label in ("N3", "N6"):
        per_window_ok = {}
        signs = []
        inconclusive = False
        for w in WINDOW_SIZES:
            t = report["windows"][str(w)][n_label]["spectral_concentration"]
            if t.get("inconclusive"):
                inconclusive = True
                per_window_ok[str(w)] = None
                continue
            supported = t["pvalue"] < ALPHA and abs(t["effect_size_r"]) >= 0.3
            per_window_ok[str(w)] = supported
            signs.append(t["effect_size_r"] > 0)

        if inconclusive:
            status = "INCONCLUSIVE"
        elif not all(per_window_ok.values()):
            status = "NOT_SUPPORTED"
        elif len(set(signs)) > 1:
            status = "NIESTABILNY"
        elif signs and signs[0] is False:
            # zgodne, istotne, ale event < baseline -- przeciwny kierunek do przewidywania
            status = "NOT_SUPPORTED (kierunek przeciwny do przewidywania)"
        else:
            status = "SUPPORTED"
        out[n_label] = {"status": status, "per_window_significant": per_window_ok}
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--time-budget", type=float, default=90.0)
    args = parser.parse_args()

    if args.extract or not (args.extract or args.analyze):
        progress = run_extraction(time_budget_s=args.time_budget)
        print(json.dumps(progress, indent=2))

    if args.analyze:
        rep = analyze()
        cls = classify(rep)
        print(json.dumps({"report": rep, "classification": cls}, indent=2))
