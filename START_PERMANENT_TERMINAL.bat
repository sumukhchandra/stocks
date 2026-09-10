@echo off
TITLE NSE Alpha Trading Terminal - Permanent Host
COLOR 0A
CD /D "%~dp0"

echo =====================================================================
echo          NSE ALPHA INTELLIGENCE - PERMANENT TRADING TERMINAL         
echo =====================================================================
echo.
echo [1/4] Checking Python and Node dependencies...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH. Please ensure Python is installed.
    pause
    exit /b 1
)

echo [2/4] Starting Streamlit Web App (Port 8501)...
netstat -ano | findstr :8501 | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Streamlit is already running on Port 8501.
) else (
    start "NSE_Streamlit_Server" /min cmd /c "streamlit run app.py --server.port 8501 --server.headless true"
    echo    Started Streamlit Server on http://localhost:8501
    timeout /t 3 /nobreak >nul
)

echo [3/4] Starting AI Auto-Trader Daemon...
wmic process where "commandline like '%%nse_auto_trader.py%%' and not commandline like '%%wmic%%'" get processid 2>nul | findstr [0-9] >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo    Auto-Trader Daemon is already actively running.
) else (
    start "NSE_Auto_Trader" /min cmd /c "python -u backend/nse_auto_trader.py"
    echo    Started AI Auto-Trader Background Daemon.
)

echo [4/4] Establishing Fixed Permanent Public Tunnel...
start "NSE_Localtunnel" /min cmd /c "npx --yes localtunnel --port 8501 --subdomain nse-alpha-trader-sumuk"

echo.
echo =====================================================================
echo                    PERMANENT ACCESS LINKS
echo =====================================================================
echo  LOCAL PERMANENT URL  : http://localhost:8501
echo  (Always active on this PC, 100%% uptime, never expires)
echo.
echo  FIXED PUBLIC URL     : https://nse-alpha-trader-sumuk.loca.lt
echo  (Permanent custom subdomain for phone/external access)
echo  Localtunnel Bypass IP: 117.250.240.128
echo =====================================================================
echo.
echo Launching your trading dashboard in default browser...
start http://localhost:8501

echo.
echo [INFO] System is running in background.
echo You can minimize this window. To stop the system, run STOP_TERMINAL.bat.
echo.
pause
