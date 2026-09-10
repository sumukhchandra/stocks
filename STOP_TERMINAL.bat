@echo off
TITLE Stop NSE Alpha Trading Terminal
COLOR 0C
CD /D "%~dp0"

echo =====================================================================
echo                STOPPING NSE ALPHA TRADING TERMINAL
echo =====================================================================
echo.

echo Stopping Streamlit Server...
taskkill /FI "WINDOWTITLE eq NSE_Streamlit_Server*" /F /T >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8501" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

echo Stopping Auto-Trader Daemon...
taskkill /FI "WINDOWTITLE eq NSE_Auto_Trader*" /F /T >nul 2>&1
wmic process where "commandline like '%%nse_auto_trader.py%%' and not commandline like '%%wmic%%'" call terminate >nul 2>&1

echo Stopping Localtunnel...
taskkill /FI "WINDOWTITLE eq NSE_Localtunnel*" /F /T >nul 2>&1

echo.
echo All NSE Trading Terminal services have been safely stopped.
echo =====================================================================
pause
