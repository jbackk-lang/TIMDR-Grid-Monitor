"""
device_client.py — szkielet pod podłączenie realnego urządzenia
================================================================================
Interfejs gotowy pod przyszłe podłączenie licznika/PLC/systemu SCADA na
żywo (Modbus, MQTT, REST). Konkretne klasy klientów (`ModbusGridClient`,
`MqttGridClient`, `RestGridClient`) są ŚWIADOMIE NIEZAIMPLEMENTOWANE -
rzucają czytelny `NotImplementedError` z instrukcją, co trzeba dopisać.
NIE udają, że działają (żadnych fałszywych/losowych odczytów pod płaszczem
"na razie tak zrobię") - to byłoby gorsze niż jawny brak implementacji,
bo wyglądałoby na działające, a analizowałoby fikcyjne dane.

`GridBufferAccumulator` NATOMIAST jest w pełni zaimplementowany i
przetestowany - to prawdziwa, użyteczna infrastruktura, z której
skorzysta KAŻDA z trzech metod podłączenia (demo/CSV/realne urządzenie):
gromadzi pojedyncze odczyty (jeden na wywołanie, tak jak przyszłby z
Modbus/MQTT/REST) w kroczącym oknie i buduje z nich `TimdrEnergySignals`
gotowe do analizy w dowolnym momencie.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from grid_monitor import TimdrEnergySignals


class GridBufferAccumulator:
    """
    Kroczący bufor pojedynczych odczytów (napięcie/częstotliwość/THD/
    obciążenie) - buduje `TimdrEnergySignals` z ostatnich `maxlen`
    próbek. Używane przez wszystkie trzy źródła danych: demo (odczyt po
    odczycie zamiast całej tablicy naraz - opcjonalnie), import CSV
    (można "odtworzyć" plik jako strumień odczytów), i - docelowo -
    realne urządzenie (Modbus/MQTT/REST wrzuca tu każdy nowy odczyt).
    """

    def __init__(self, maxlen: int = 6000):
        self.maxlen = maxlen
        self._voltage: deque = deque(maxlen=maxlen)
        self._frequency: deque = deque(maxlen=maxlen)
        self._harmonics: deque = deque(maxlen=maxlen)
        self._load: deque = deque(maxlen=maxlen)
        self._timestamps: deque = deque(maxlen=maxlen)

    def push(self, voltage: float, frequency: float, harmonics: float, load: float, timestamp=None) -> None:
        self._voltage.append(float(voltage))
        self._frequency.append(float(frequency))
        self._harmonics.append(float(harmonics))
        self._load.append(float(load))
        self._timestamps.append(timestamp)

    def __len__(self) -> int:
        return len(self._voltage)

    def is_ready(self, min_samples: int = 8) -> bool:
        """`rhythm()`/`resonance()` potrzebują minimalnej liczby próbek,
        żeby wynik miał sens - sprawdź to przed wywołaniem `to_signals()`
        w pętli na żywo, żeby nie analizować pustego/zbyt krótkiego okna."""
        return len(self) >= min_samples

    def to_signals(self) -> TimdrEnergySignals:
        if len(self) == 0:
            raise ValueError("GridBufferAccumulator: bufor jest pusty - brak odczytów do analizy.")
        return TimdrEnergySignals(
            voltage=np.array(self._voltage, dtype=float),
            frequency=np.array(self._frequency, dtype=float),
            harmonics=np.array(self._harmonics, dtype=float),
            load=np.array(self._load, dtype=float),
        )

    def clear(self) -> None:
        self._voltage.clear()
        self._frequency.clear()
        self._harmonics.clear()
        self._load.clear()
        self._timestamps.clear()


class GridDeviceClient:
    """Wspólny interfejs dla klientów realnych urządzeń. Konkretna
    implementacja MUSI dostarczyć `read_once()` - jeden odczyt czterech
    wartości w danej chwili. `poll_into(accumulator, n_readings)` jest
    już gotowe i przetestowane - działa z DOWOLNĄ konkretną
    implementacją `read_once()`, więc podłączenie nowego protokołu to
    tylko dopisanie jednej metody, reszta (buforowanie, budowa
    TimdrEnergySignals) już działa."""

    def read_once(self) -> tuple[float, float, float, float]:
        """Zwraca (voltage, frequency, harmonics_thd, load) - JEDEN
        odczyt. Musi być nadpisane przez konkretną implementację."""
        raise NotImplementedError(
            f"{type(self).__name__}.read_once() nie jest zaimplementowane - "
            "to jest szkielet interfejsu, nie działający klient. Zobacz "
            "docstring klasy, żeby dowiedzieć się, co dopisać dla Twojego "
            "urządzenia."
        )

    def poll_into(self, accumulator: GridBufferAccumulator, n_readings: int = 1) -> None:
        for _ in range(n_readings):
            v, f, h, load = self.read_once()
            accumulator.push(v, f, h, load)


class ModbusGridClient(GridDeviceClient):
    """
    SZKIELET - niezaimplementowane. Żeby podłączyć realny licznik po
    Modbus TCP/RTU:

      1. `pip install pymodbus`
      2. Ustal mapę rejestrów SWOJEGO konkretnego licznika (różni
         producenci - Janitza, Schneider, Carlo Gavazzi - mają różne
         adresy rejestrów dla napięcia/częstotliwości/THD/mocy; ta
         informacja jest w dokumentacji urządzenia, nie da się jej
         zgadnąć).
      3. W `read_once()` odpytaj rejestry (`client.read_holding_registers`
         albo `read_input_registers`, zależnie od urządzenia) i przelicz
         surowe wartości rejestrów na jednostki fizyczne wg dokumentacji
         (często wymaga przemnożenia przez współczynnik skalujący z
         innego rejestru).

    Parametry konstruktora (host/port/unit_id) są tu jako przykład
    typowego kształtu API pymodbus - nieużywane, dopóki `read_once()`
    nie zostanie zaimplementowane.
    """

    def __init__(self, host: str, port: int = 502, unit_id: int = 1):
        self.host = host
        self.port = port
        self.unit_id = unit_id


class MqttGridClient(GridDeviceClient):
    """
    SZKIELET - niezaimplementowane. Żeby podłączyć urządzenie
    publikujące odczyty przez MQTT:

      1. `pip install paho-mqtt`
      2. Zasubskrybuj właściwy topic (`client.subscribe(topic)`) i w
         callbacku `on_message` sparsuj payload (JSON? CSV? zależy od
         urządzenia) do (voltage, frequency, harmonics, load).
      3. `read_once()` w typowym MQTT-owym modelu "push" nie pasuje
         idealnie (broker wysyła dane, gdy chce, nie na żądanie) -
         praktyczna implementacja zwykle trzyma "ostatnią otrzymaną
         wiadomość" w buforze wewnętrznym i `read_once()` ją zwraca,
         blokując/czekając jeśli jeszcze nic nie przyszło.
    """

    def __init__(self, broker_host: str, topic: str, port: int = 1883):
        self.broker_host = broker_host
        self.topic = topic
        self.port = port


class RestGridClient(GridDeviceClient):
    """
    SZKIELET - niezaimplementowane. Żeby podłączyć urządzenie/system
    SCADA wystawiające dane przez REST API:

      1. `pip install requests`
      2. W `read_once()` wykonaj `requests.get(self.url, ...)`,
         sparsuj JSON odpowiedzi wg schematu KONKRETNEGO API (nazwy pól
         różnią się między systemami - to samo ostrzeżenie co dla
         Modbus: nie da się zgadnąć, trzeba znać dokumentację API).
      3. Obsłuż uwierzytelnianie, jeśli API tego wymaga (token/klucz) -
         NIGDY nie zapisuj sekretów na stałe w kodzie źródłowym.
    """

    def __init__(self, url: str, auth_token: str | None = None):
        self.url = url
        self.auth_token = auth_token
