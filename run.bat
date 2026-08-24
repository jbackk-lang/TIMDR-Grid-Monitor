@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo  TIMDR-Grid-Monitor - lokalne API + dashboard
echo  UWAGA: narzedzie badawczo-edukacyjne, NIE zastepuje
echo  certyfikowanego analizatora jakosci energii. Patrz README.md.
echo ============================================================
echo.

rem --- Znajdz dzialajace polecenie Pythona (pip samo w sobie moze nie byc
rem     na PATH, nawet gdy Python jest zainstalowany - dlatego wolamy
rem     zawsze "%PYCMD% -m pip", NIE samo "pip") ---
set "PYCMD="

python --version >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=python"
    goto :found_python
)

py --version >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py"
    goto :found_python
)

echo BLAD: nie znaleziono Pythona (polecenia "python" ani "py" nie dzialaja).
echo.
echo Zainstaluj Pythona z https://www.python.org/downloads/
echo WAZNE: podczas instalacji zaznacz checkbox "Add python.exe to PATH"
echo (jest na pierwszym ekranie instalatora, na dole).
echo.
echo Po instalacji zamknij to okno i uruchom run.bat ponownie
echo (moze byc potrzebny restart terminala/eksploratora, zeby PATH sie odswiezyl).
pause
exit /b 1

:found_python
echo Uzywam interpretera: %PYCMD%
%PYCMD% --version

echo.
echo Sprawdzam pip...
%PYCMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo BLAD: %PYCMD% jest zainstalowany, ale modul pip nie dziala.
    echo Sprobuj: %PYCMD% -m ensurepip --upgrade
    pause
    exit /b 1
)

echo.
echo Instalacja/aktualizacja zaleznosci...
%PYCMD% -m pip install --quiet flask numpy pandas openpyxl pytest
if errorlevel 1 (
    echo BLAD: nie udalo sie zainstalowac zaleznosci pip.
    echo Sprawdz polaczenie z internetem albo uruchom recznie:
    echo   %PYCMD% -m pip install flask numpy pandas openpyxl pytest
    pause
    exit /b 1
)

echo.
echo Uruchamiam testy (pytest)...
%PYCMD% -m pytest -q
if errorlevel 1 (
    echo.
    echo UWAGA: co najmniej jeden test nie przeszedl. Serwer uruchomi
    echo sie mimo to, ale sprawdz powyzsze wyniki testow.
    echo.
)

echo.
echo Start serwera na http://127.0.0.1:8070
echo (port 5060 i kilka innych celowo NIE sa uzywane - sa na liscie
echo  "zakazanych portow" przegladarek/fetch(), patrz README.md)
echo (Ctrl+C aby zatrzymac)
echo.
start "" http://127.0.0.1:8070
%PYCMD% api.py

pause
