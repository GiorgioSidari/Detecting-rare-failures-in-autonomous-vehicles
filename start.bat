@echo off
REM ============================================================
REM  Rare Failure Detection — AV Safety
REM  Avvia il server FastAPI con il venv del progetto
REM ============================================================

cd /d "%~dp0"

REM Attiva il virtual environment
call .venv\Scripts\activate.bat

REM Controlla che uvicorn sia disponibile
where uvicorn >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [ERRORE] uvicorn non trovato nel venv.
    echo  Esegui prima:  pip install -e ".[dev]"  oppure  pip install fastapi uvicorn[standard]
    echo.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Rare Failure Detection API
echo   http://localhost:8001
echo   Frontend: apri frontend/index.html nel browser
echo  ============================================================
echo.

uvicorn api.server:app --host 0.0.0.0 --port 8001 --reload

pause
