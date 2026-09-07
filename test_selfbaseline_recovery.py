"""
test_selfbaseline_recovery.py -- ten sam test co w calej rodzinie repo TIMDR
(TIMDR-Crypto-Graph, universal-state-analyzer, deliverable_timdr_finanse,
analizator-gieldowy-v3, TIMDR-Earthquake-Core): czy anomalies()/defect()
falszywie flaguja NOWE, normalne probki po ustaniu anomalii sieciowej,
tylko dlatego ze stare anomalne probki wciaz siedza w oknie referencyjnym?

WYNIK (brak bledu - w przeciwienstwie do deliverable_timdr_finanse, gdzie
ten sam typ testu znalazl realny bug w defekt()):

- defect() ma WBUDOWANE, przyczynowe okno kroczace (window=20, patrz
  _rolling_percentile_spread) na rozrzucie SAMYCH ROZNIC (nie poziomow -
  ten blad juz zaadresowany w projekcie tego pliku). Kontaminacja
  wypada z okna automatycznie po `window` probkach - sprawdzone: ZERO
  falszywych flag po tym jak anomalia opuszcza okno, na 5 ziarnach.
- anomalies() (MAD-z, self-baseline) nie ma wbudowanego okna - w trybie
  strumieniowym (trailing window W=30) odzysk jest niemal natychmiastowy
  dzieki odpornosci mediany/MAD na mniejszosciowa kontaminacje (ten sam
  wzorzec co w universal-state-analyzer/deliverable_timdr_finanse).
"""
import numpy as np

from grid_core import anomalies, defect


def _mad_z_of_last(window_vals):
    x = np.asarray(window_vals, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 1e-12:
        scale = 1.4826 * mad
    else:
        spread = np.max(x) - np.min(x)
        scale = spread / 4.0 if spread > 1e-12 else 1.0
    return (x[-1] - med) / scale


def test_defect_zero_falszywych_flag_po_opuszczeniu_okna():
    """defect() ma wbudowane okno=20 - po tym jak anomalia calkowicie z
    niego wypadnie, kolejne normalne probki NIE powinny byc flagowane."""
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n_pre, n_anom, n_post = 100, 5, 150
        full = np.concatenate([
            rng.normal(230, 0.5, n_pre),   # normalne napiecie ~230V
            np.full(n_anom, 260.0),         # anomalia: skok napiecia
            rng.normal(230, 0.5, n_post),
        ])
        idx = defect(full, window=20, jump_factor=3.0)
        event_end = n_pre + n_anom
        post_flags = idx[idx >= event_end + 20]
        assert len(post_flags) == 0, (
            f"seed={seed}: defect() wciaz flaguje probki po opuszczeniu "
            f"okna anomalii: {list(post_flags)}"
        )


def test_anomalies_recovers_szybko_strumieniowo():
    """anomalies() bez wbudowanego okna - w trybie strumieniowym (trailing
    window W=30) probki tuz po evencie nie powinny byc systematycznie
    odstajace (|z| < 3.0, prog domyslny anomalies())."""
    W = 30
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n_pre, n_anom = 60, 3
        full = np.concatenate([
            rng.normal(230, 0.5, n_pre),
            np.full(n_anom, 260.0),
            rng.normal(230, 0.5, 20),
        ])
        event_end = n_pre + n_anom
        post_z = [
            _mad_z_of_last(full[max(0, i - W):i + 1])
            for i in range(event_end, event_end + 5)
        ]
        assert all(abs(z) < 3.0 for z in post_z), (
            f"seed={seed}: anomalies() flaguje probki tuz po evencie ({post_z})"
        )
