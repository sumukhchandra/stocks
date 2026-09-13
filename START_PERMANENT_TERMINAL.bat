@echo off
TITLE NSE Alpha Trading Terminal - Permanent Host
COLOR 0A
CD /D "%~dp0"

echo =====================================================================
echo          NSE ALPHA INTELLIGENCE - DECOUPLED TRADING TERMINAL         
echo =====================================================================
echo.
echo [1/4] Checking Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH. Please ensure Python is installed.
    pause
    exit /b 1
)

echo [2/4] Starting FastAPI High-Performance Backend API (Port 8000)...
netstat -ano | findstr :8000 | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Backend API is already running on Port 8000.
) else (
    start "NSE_FastAPI_Backend" /min cmd /c "python -m uvicorn apps.api.src.main:app --host 0.0.0.0 --port 8000"
    echo    Started Backend API Server on http://0.0.0.0:8000
    timeout /t 3 /nobreak >nul
)

echo [3/4] Starting Streamlit Web Terminal (Port 8501)...
netstat -ano | findstr :8501 | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Streamlit is already running on Port 8501.
) else (
    start "NSE_Streamlit_Server" /min cmd /c "streamlit run frontend/app.py --server.port 8501 --server.headless true"
    echo    Started Streamlit Server on http://localhost:8501
    timeout /t 2 /nobreak >nul
)

echo [4/4] Starting Public API Tunnel for Mobile App (Port 8000)...
start "NSE_Localtunnel_Backend" /min cmd /c "npx --yes localtunnel --port 8000 --subdomain nse-alpha-backend-sumuk"

echo.
echo =====================================================================
echo                    ACCESS LINKS & CONFIGURATION
echo =====================================================================
echo  LOCAL BACKEND API     : http://localhost:8000
echo  MOBILE WEB APP        : http://localhost:8000/mobile/
echo  DESKTOP STREAMLIT APP : http://localhost:8501
echo.
echo  ONLINE PUBLIC API URL : https://nse-alpha-backend-sumuk.loca.lt
echo  API KEY AUTHENTICATION: nse_secret_alpha_2026
echo =====================================================================
echo.
echo Launching Mobile App and Terminal in default browser...
start http://localhost:8000/mobile/
start http://localhost:8501

echo.
echo [INFO] All services are actively running.
echo To stop everything safely, run STOP_TERMINAL.bat.
echo.
pause
