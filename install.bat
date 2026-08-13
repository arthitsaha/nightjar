@echo off
REM One-shot setup for Nightjar on Windows.
cd /d "%~dp0"
where py >nul 2>&1
if %errorlevel%==0 (
    py -3 install.py %*
) else (
    python install.py %*
)
pause
