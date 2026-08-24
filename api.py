"""
api.py — TIMDR-Grid-Monitor, lokalne REST API + dashboard
================================================================================
Pojedynczy proces Flask (ten sam wzorzec co inne dashboardy TIMDR w tym
zestawie repo) - serwuje `/` (dashboard) oraz endpointy JSON.

UWAGA nt. portu: 8070 wybrany celowo - port 5060 (i kilka innych, np.
5061/6000/6666-6669) jest na liście "zakazanych portów" przeglądarek i
fetch() (Node/undici) - dashboard uruchomiony na takim porcie nigdy nie
połączyłby się z własnym API w realnej przeglądarce. Ta pułapka została
znaleziona i udokumentowana przy budowie poprzedniego repo w tym
zestawie (analizator-gieldowy-v3) - tutaj zastosowana od razu.

Endpointy:
  GET  /                  -> dashboard
  GET  /api/health
  GET  /api/scenarios     -> lista dostępnych scenariuszy demo
  GET  /api/demo          -> analiza scenariusza syntetycznego (?scenario=...&rated_load=...&<cable_params>)
  POST /api/csv           -> analiza wgranego pliku CSV/Excel (multipart/form-data, + <cable_params>)

Oba endpointy analizy (`/api/demo`, `/api/csv`) dodatkowo zwracają:
  - `forecast` - prognoza kolejnego odczytu per kanał (model Holta, patrz
    forecast_core.py) + interpretacja przewidywanych zdarzeń.
  - `cable_life` - szacowana żywotność kabla/linii pod danym profilem
    obciążenia (patrz cable_life.py). Parametry kabla (opcjonalne, z
    sensownymi domyślnymi): `insulation_type` (pvc/xlpe/epr),
    `conductor_material` (copper/aluminum), `ambient_temp_c`,
    `years_in_service`, `thermal_halving_deltaT_c`.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from grid_monitor import TimdrEnergySignals, TimdrEnergyEvents, TimdrEnergyMonitor
from demo_generator import generate as generate_demo, SCENARIOS, SAMPLE_RATE_HZ, DEFAULT_RATED_LOAD_W, V_NOMINAL, F_NOMINAL
from csv_loader import load_csv, CsvLoaderError
from forecast_core import TimdrEnergyPredictor, TimdrEnergyForecastEvents, TimdrEnergyForecaster
from cable_life import CableSpec, estimate_remaining_life

# Parametry CableSpec, które wolno nadpisać z żądania HTTP (nazwa -> typ konwersji)
_CABLE_PARAM_TYPES = {
    "insulation_type": str,
    "conductor_material": str,
    "ambient_temp_c": float,
    "years_in_service": float,
    "thermal_halving_deltaT_c": float,
    "design_life_years": float,
}

app = Flask(__name__, static_folder="static", static_url_path="")

DISCLAIMER = (
    "Narzędzie badawczo-edukacyjne do demonstracji wykrywania zdarzeń w sieci "
    "energetycznej metodą TIMDR. NIE zastępuje certyfikowanego analizatora "
    "jakości energii ani nie jest urządzeniem pomiarowym zgodnym z normą "
    "IEC 61000-4-30. Progi oparte o typowe wartości EN 50160 - przed użyciem "
    "produkcyjnym dostosuj do parametrów własnej instalacji (moc znamionowa, "
    "lokalne wymagania). Prognoza (`forecast`) to prosta ekstrapolacja "
    "statystyczna (model Holta), nie sieć AI wytrenowana na realnych danych. "
    "Szacunek żywotności kabla (`cable_life`) to uproszczony model inżynierski "
    "(grzanie I²R + reguła Montsingera/Arrheniusa dla starzenia izolacji) - "
    "pokazuje TREND pod danym profilem obciążenia, nie certyfikowaną ocenę "
    "wg IEC 60287/IEC 60216."
)

SCENARIO_LABELS = {
    "normalny": "Typowa praca sieci (bez istotnych zdarzeń)",
    "przeciazenie": "Przeciążenie",
    "mikrozanik": "Mikro-zaniki napięcia",
    "anomalia_harmoniczna": "Anomalia harmoniczna (THD)",
    "cykliczne_zaklocenia": "Cykliczne zakłócenia obciążenia",
    "mieszany": "Scenariusz mieszany (wszystkie typy)",
}


def _clean(obj):
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _clean(float(obj))
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def _events_to_dict(events: TimdrEnergyEvents) -> dict:
    return {
        "overloads": [{"idx": i, "load": v, "rated_load": r} for i, v, r in events.overloads],
        "micro_outages": [{"start_idx": s, "duration_ms": d} for s, d in events.micro_outages],
        "harmonic_anomalies": [{"idx": i, "thd": v, "powod": p} for i, v, p in events.harmonic_anomalies],
        "frequency_anomalies": [{"idx": i, "freq": v, "poziom": p} for i, v, p in events.frequency_anomalies],
        "cyclic_disturbances": [{"channel": ch, "period_samples": p, "power": pw} for ch, p, pw in events.cyclic_disturbances],
    }


def _parse_cable_params(source) -> dict:
    """Wyciąga opcjonalne parametry CableSpec z request.args/request.form.
    Rzuca ValueError z czytelnym komunikatem przy niepoprawnej liczbie -
    nigdy nie ignoruje cicho błędnego wejścia użytkownika."""
    params = {}
    for name, caster in _CABLE_PARAM_TYPES.items():
        raw = source.get(name)
        if raw is None or raw == "":
            continue
        try:
            params[name] = caster(raw)
        except ValueError:
            raise ValueError(f"Parametr kabla '{name}' ma niepoprawną wartość: '{raw}'")
    return params


def _forecast_events_to_dict(events: TimdrEnergyForecastEvents) -> dict:
    return {
        "predicted_overload": events.predicted_overload,
        "predicted_micro_outage": events.predicted_micro_outage,
        "predicted_harmonic_spike": events.predicted_harmonic_spike,
    }


def _signals_to_dict(signals: TimdrEnergySignals) -> dict:
    return {
        "voltage": signals.voltage.tolist(),
        "frequency": signals.frequency.tolist(),
        "harmonics": signals.harmonics.tolist(),
        "load": signals.load.tolist(),
    }


def _summary(events: TimdrEnergyEvents, n_samples: int, sample_rate_hz: float) -> dict:
    n_events = (
        len(events.overloads) + len(events.micro_outages)
        + len(events.harmonic_anomalies) + len(events.frequency_anomalies)
    )

    # Rozróżnienie ALARM vs UWAGA: przeciążenia i mikro-zaniki to zawsze
    # ALARM (bezpośrednie zagrożenie), tak samo przekroczenie NORMY THD
    # (EN 50160, 8%) i "krytyczne" odchylenie częstotliwości (>4%). Same
    # w sobie flagi WYŁĄCZNIE adaptacyjne (MAD-z, bez przekroczenia
    # twardego limitu) mają nieuniknioną, statystyczną częstość
    # fałszywych trafień przy dużej liczbie próbek (np. ~0.2% dla
    # factor=3 na w przybliżeniu normalnym rozkładzie) - pojedyncze takie
    # flagi na typowej pracy sieci to sygnał do obserwacji (UWAGA), nie
    # potwierdzone zagrożenie (ALARM). Patrz README.md.
    harmonic_alarm = any(p in ("limit_normy", "oba") for _, _, p in events.harmonic_anomalies)
    harmonic_watch = any(p == "odchylenie_adaptacyjne" for _, _, p in events.harmonic_anomalies)
    freq_alarm = any(p == "krytyczne" for _, _, p in events.frequency_anomalies)
    freq_watch = any(p == "normalne_odchylenie" for _, _, p in events.frequency_anomalies)

    if events.overloads or events.micro_outages or harmonic_alarm or freq_alarm:
        status = "ALARM"
    elif harmonic_watch or freq_watch or events.cyclic_disturbances:
        status = "UWAGA"
    else:
        status = "OK"
    return {
        "status": status,
        "n_samples": n_samples,
        "duration_s": round(n_samples / sample_rate_hz, 2) if sample_rate_hz else None,
        "n_overloads": len(events.overloads),
        "n_micro_outages": len(events.micro_outages),
        "n_harmonic_anomalies": len(events.harmonic_anomalies),
        "n_frequency_anomalies": len(events.frequency_anomalies),
        "n_cyclic_disturbances": len(events.cyclic_disturbances),
        "n_total_events": n_events,
    }


def _run_analysis(signals: TimdrEnergySignals, sample_rate_hz: float, rated_load: float, cable_params: dict | None = None) -> dict:
    monitor = TimdrEnergyMonitor(v_nominal=V_NOMINAL, f_nominal=F_NOMINAL, rated_load=rated_load)
    events = monitor.analyze(signals, sample_rate_hz)

    predictor = TimdrEnergyPredictor()
    prediction = predictor.predict_next(signals.voltage, signals.frequency, signals.load, signals.harmonics)
    forecaster = TimdrEnergyForecaster(rated_load=rated_load, v_nominal=V_NOMINAL, f_nominal=F_NOMINAL)
    forecast_events = forecaster.analyze_prediction(prediction, recent_harmonics=signals.harmonics)

    spec = CableSpec(rated_load_w=rated_load, **(cable_params or {}))
    cable_life = estimate_remaining_life(signals.load, sample_rate_hz, spec)

    result = {
        "sample_rate_hz": sample_rate_hz,
        "rated_load": rated_load,
        "v_nominal": V_NOMINAL,
        "f_nominal": F_NOMINAL,
        "signals": _signals_to_dict(signals),
        "events": _events_to_dict(events),
        "summary": _summary(events, len(signals.load), sample_rate_hz),
        "forecast": {
            "prediction": prediction,
            "events": _forecast_events_to_dict(forecast_events),
        },
        "cable_life": cable_life,
        "disclaimer": DISCLAIMER,
    }
    return result


@app.route("/")
def dashboard():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "disclaimer": DISCLAIMER})


@app.route("/api/scenarios")
def scenarios():
    return jsonify({"scenarios": [{"id": k, "label": SCENARIO_LABELS.get(k, k)} for k in SCENARIOS]})


@app.route("/api/demo")
def demo():
    scenario = request.args.get("scenario", "normalny")
    try:
        rated_load = float(request.args.get("rated_load", DEFAULT_RATED_LOAD_W))
    except ValueError:
        return jsonify({"error": "rated_load musi być liczbą"}), 400

    if scenario not in SCENARIOS:
        return jsonify({"error": f"Nieznany scenariusz '{scenario}'. Dostępne: {sorted(SCENARIOS)}"}), 400

    try:
        cable_params = _parse_cable_params(request.args)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    signals = generate_demo(scenario)
    try:
        result = _run_analysis(signals, SAMPLE_RATE_HZ, rated_load, cable_params)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    result["scenario"] = scenario
    return jsonify(_clean(result))


@app.route("/api/csv", methods=["POST"])
def csv_analyze():
    if "file" not in request.files:
        return jsonify({"error": "Brak pliku w żądaniu (pole 'file')"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Nie wybrano pliku"}), 400

    try:
        rated_load = float(request.form.get("rated_load", DEFAULT_RATED_LOAD_W))
    except ValueError:
        return jsonify({"error": "rated_load musi być liczbą"}), 400
    load_unit = request.form.get("load_unit", "W")

    try:
        cable_params = _parse_cable_params(request.form)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    suffix = os.path.splitext(f.filename)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        signals, sample_rate_hz = load_csv(tmp_path, load_unit=load_unit)
    except CsvLoaderError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        try:
            os.remove(tmp_path)
        except OSError as e:
            print(f"[api.py] UWAGA: nie udało się usunąć pliku tymczasowego '{tmp_path}': {e}")

    try:
        result = _run_analysis(signals, sample_rate_hz, rated_load, cable_params)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    result["source_file"] = f.filename
    return jsonify(_clean(result))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8070, debug=False, threaded=True)
