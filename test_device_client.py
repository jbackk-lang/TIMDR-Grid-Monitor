"""test_device_client.py — testy dla device_client.py."""

import numpy as np
import pytest

from device_client import (
    GridBufferAccumulator, GridDeviceClient,
    ModbusGridClient, MqttGridClient, RestGridClient,
)


def test_accumulator_pusty_na_starcie():
    acc = GridBufferAccumulator(maxlen=100)
    assert len(acc) == 0
    assert not acc.is_ready()


def test_accumulator_push_i_to_signals():
    acc = GridBufferAccumulator(maxlen=100)
    for i in range(20):
        acc.push(voltage=230.0 + i * 0.01, frequency=50.0, harmonics=2.0, load=2000.0)
    assert len(acc) == 20
    assert acc.is_ready()
    signals = acc.to_signals()
    assert len(signals.voltage) == 20
    assert signals.voltage[0] == pytest.approx(230.0)
    assert signals.voltage[-1] == pytest.approx(230.19)


def test_accumulator_respektuje_maxlen_jako_kroczace_okno():
    acc = GridBufferAccumulator(maxlen=10)
    for i in range(25):
        acc.push(voltage=float(i), frequency=50.0, harmonics=2.0, load=2000.0)
    assert len(acc) == 10
    signals = acc.to_signals()
    # powinny zostac tylko OSTATNIE 10 odczytow (15..24)
    assert list(signals.voltage) == [float(x) for x in range(15, 25)]


def test_accumulator_pusty_to_signals_rzuca_czytelny_blad():
    acc = GridBufferAccumulator()
    with pytest.raises(ValueError, match="pusty"):
        acc.to_signals()


def test_accumulator_clear():
    acc = GridBufferAccumulator()
    acc.push(230.0, 50.0, 2.0, 2000.0)
    acc.clear()
    assert len(acc) == 0


# ---------------------------------------------------------------------
# Szkielety klientów - MUSZĄ jawnie zawodzić, nigdy cicho "udawać" danych
# ---------------------------------------------------------------------

@pytest.mark.parametrize("client_cls,kwargs", [
    (ModbusGridClient, {"host": "192.168.1.10"}),
    (MqttGridClient, {"broker_host": "mqtt.local", "topic": "grid/reading"}),
    (RestGridClient, {"url": "http://device.local/api/reading"}),
])
def test_szkielet_klienta_rzuca_not_implemented_zamiast_udawac(client_cls, kwargs):
    client = client_cls(**kwargs)
    with pytest.raises(NotImplementedError, match="read_once"):
        client.read_once()


def test_poll_into_dziala_z_dowolna_implementacja_read_once():
    """`poll_into` jest gotowe niezależnie od konkretnego protokołu -
    weryfikacja na prostej, ręcznej implementacji `read_once()`."""

    class FakeClient(GridDeviceClient):
        def __init__(self):
            self.calls = 0

        def read_once(self):
            self.calls += 1
            return (230.0, 50.0, 2.0, 2000.0 + self.calls)

    client = FakeClient()
    acc = GridBufferAccumulator()
    client.poll_into(acc, n_readings=5)
    assert len(acc) == 5
    assert client.calls == 5
    signals = acc.to_signals()
    assert list(signals.load) == [2001.0, 2002.0, 2003.0, 2004.0, 2005.0]
