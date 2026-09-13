@echo off
cd /d "%~dp0"

REM Launch pythonw in the background without keeping this terminal window open
if exist "C:\Users\sumuk\AppData\Local\Programs\Python\Python311\pythonw.exe" (
    start "" "C:\Users\sumuk\AppData\Local\Programs\Python\Python311\pythonw.exe" run_desktop_app.py
) else (
    start "" pythonw run_desktop_app.py
)

exit
