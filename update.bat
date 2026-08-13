@echo off
REM Pull the latest Nightjar. Works with or without git installed.
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (
    py -3 update.py %*
) else (
    python update.py %*
)
pause
