# B4-Grid / PROTECT-90 — otwarty plan danych

To jest nowy tor danych. Nie zmienia wcześniejszych demo, testów ani wyników.

Źródło: PROTECT-90 v1.0.0, DOI 10.5281/zenodo.21109169, CC BY 4.0.
Zbiór ma 9 022 jednosekundowe, zsynchronizowane przebiegi napięcia i prądu
próbkowane 6.4 kHz. Są to fizycznie ugruntowane symulacje EMT zwarć w sieci
90 kV, nie rejestracje terenowe.

## Stan

Manifest data/protect90/B4_GRID_PROTECT90_OPEN_MANIFEST.json jest otwarty
wyłącznie na pobranie i sprawdzenie danych. Nie wolno jeszcze publikować wyniku B4.

## Pobranie

    py -3.12 download_protect90.py
    py -3.12 download_protect90.py --full

Druga komenda pobiera około 12 GB. Skrypt sprawdza opublikowane sumy MD5.

## Zamrożenie przed analizą

1. Zapisać SHA-256 pobranego archiwum i CSV.
2. Wybrać listę sample_id bez oglądania przebiegów oraz zapisać ją w repo.
3. Ustalić jedną lokalizację pomiarową i podział epizodowy.
4. Utworzyć manifest FROZEN z listą oraz hashami.

Adapter protect90_adapter.py jawnie redukuje przebiegi trójfazowe do czterech
serii Grid Monitor. Load jest proxy mocy pozornej, a nie zmierzoną mocą czynną.
