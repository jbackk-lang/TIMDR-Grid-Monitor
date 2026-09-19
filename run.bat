@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
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
echo Sprawdzam lokalne zaleznosci...
%PYCMD% -c "import flask, numpy, pandas, openpyxl, pytest" >nul 2>&1
if not errorlevel 1 goto :dependencies_ready

echo Brakuje zaleznosci. Instaluje je jednorazowo...
%PYCMD% -m pip install flask numpy pandas openpyxl pytest
if errorlevel 1 goto :dependencies_failed
goto :dependencies_ready

:dependencies_failed
echo BLAD: nie udalo sie zainstalowac zaleznosci pip.
echo Sprawdz polaczenie z internetem albo uruchom recznie:
echo   %PYCMD% -m pip install flask numpy pandas openpyxl pytest
pause
exit /b 1

:dependencies_ready

echo.
echo Tryb natychmiastowy: pomijam testy przy starcie.
echo Pelna walidacja jest dostepna po uruchomieniu z dashboardu
echo lub recznie: run.bat --full-tests
if /I not "%~1"=="--full-tests" goto :start_server
echo Uruchamiam pelny zestaw testow. To moze potrwac kilka minut.
rem --- Wlasny katalog tymczasowy omija zablokowany/uszkodzony Temp Windows ---
%PYCMD% -m pytest -q --basetemp=".pytest_tmp"
if not errorlevel 1 goto :start_server
echo.
echo UWAGA: co najmniej jeden test nie przeszedl. Serwer uruchomi sie mimo to.
echo.

:start_server
echo.
echo Start serwera na http://127.0.0.1:8070
echo (port 5060 i kilka innych celowo NIE sa uzywane - sa na liscie
echo  "zakazanych portow" przegladarek/fetch(), patrz README.md)
echo (Ctrl+C aby zatrzymac)
echo.
start "" http://127.0.0.1:8070
rem --- CALL zachowuje sterowanie w tym pliku nawet, gdy "python" na danym
rem     komputerze jest skryptem .bat zamiast bezposrednim python.exe. ---
call %PYCMD% api.py
set "SERVER_EXIT=%ERRORLEVEL%"

echo.
if "%SERVER_EXIT%"=="0" goto :server_stopped
echo [BLAD] Serwer zakonczyl dzialanie z kodem %SERVER_EXIT%.
echo Tresc bledu znajduje sie powyzej.
goto :after_server

:server_stopped
echo Serwer zostal zatrzymany.

:after_server
echo Nacisnij dowolny klawisz, aby zamknac to okno.
pause >nul
exit /b %SERVER_EXIT%
