"""test_api.py — testy Flask API (api.py) przez test_client, bez
uruchamiania prawdziwego serwera sieciowego."""

import io

import numpy as np
import pandas as pd
import pytest

import api


@pytest.fixture
def client():
    api.app.config["TESTING"] = True
    with api.app.test_client() as c:
        yield c


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_scenarios_lista(client):
    res = client.get("/api/scenarios")
    data = res.get_json()
    ids = {s["id"] for s in data["scenarios"]}
    assert ids == {"normalny", "przeciazenie", "mikrozanik", "anomalia_harmoniczna", "cykliczne_zaklocenia", "mieszany"}


def test_demo_dashboard_serwowany(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"TIMDR-Grid-Monitor" in res.data


def test_demo_normalny_status_nie_alarm(client):
    """Regresja: scenariusz 'normalny' generował ~12 statystycznych flag
    adaptacyjnych THD (oczekiwana częstość fałszywych trafień progu
    MAD-z na dużej próbce, nie prawdziwa anomalia - żadna nie przekracza
    normy EN50160 8%) - status nie powinien eskalować do ALARM za same
    flagi adaptacyjne bez przekroczenia twardego limitu."""
    res = client.get("/api/demo?scenario=normalny&rated_load=10000")
    data = res.get_json()
    assert res.status_code == 200
    assert data["summary"]["status"] in ("OK", "UWAGA")
    assert data["summary"]["n_overloads"] == 0
    assert data["summary"]["n_micro_outages"] == 0


def test_demo_przeciazenie_daje_alarm(client):
    res = client.get("/api/demo?scenario=przeciazenie&rated_load=10000")
    data = res.get_json()
    assert data["summary"]["status"] == "ALARM"
    assert data["summary"]["n_overloads"] > 0


def test_demo_mikrozanik_daje_alarm(client):
    res = client.get("/api/demo?scenario=mikrozanik&rated_load=10000")
    data = res.get_json()
    assert data["summary"]["status"] == "ALARM"
    assert data["summary"]["n_micro_outages"] > 0


def test_demo_zwraca_sygnaly_i_dyscalimer(client):
    res = client.get("/api/demo?scenario=mieszany&rated_load=10000")
    data = res.get_json()
    assert set(data["signals"].keys()) == {"voltage", "frequency", "harmonics", "load"}
    assert "disclaimer" in data and len(data["disclaimer"]) > 20


def test_demo_nieznany_scenariusz_400(client):
    res = client.get("/api/demo?scenario=nieistniejacy")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_demo_niepoprawny_rated_load_400(client):
    res = client.get("/api/demo?scenario=normalny&rated_load=abc")
    assert res.status_code == 400


def _make_csv_bytes(n=300):
    rng = np.random.default_rng(0)
    ts = pd.date_range("2026-01-01", periods=n, freq=pd.Timedelta(seconds=1))
    df = pd.DataFrame({
        "timestamp": ts,
        "voltage": 230.0 + rng.normal(0, 0.5, n),
        "frequency": 50.0 + rng.normal(0, 0.02, n),
        "THD_U": np.abs(rng.normal(2.0, 0.3, n)),
        "load": 2000.0 + rng.normal(0, 50, n),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def test_csv_brak_pliku_400(client):
    res = client.post("/api/csv", data={"rated_load": "5000"})
    assert res.status_code == 400


def test_csv_poprawny_upload_200(client):
    buf = _make_csv_bytes()
    res = client.post(
        "/api/csv",
        data={"file": (buf, "grid.csv"), "rated_load": "5000", "load_unit": "W"},
        content_type="multipart/form-data",
    )
    data = res.get_json()
    assert res.status_code == 200
    assert data["summary"]["n_samples"] == 300
    assert data["source_file"] == "grid.csv"


def test_csv_brakujaca_kolumna_daje_czytelny_400(client):
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq=pd.Timedelta(seconds=1)),
        "frequency": 50.0 + rng.normal(0, 0.02, n),
        "THD_U": np.abs(rng.normal(2.0, 0.3, n)),
        "load": 2000.0 + rng.normal(0, 50, n),
    })  # brak kolumny voltage
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    res = client.post(
        "/api/csv",
        data={"file": (buf, "bad.csv"), "rated_load": "5000"},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "voltage" in res.get_json()["error"]
