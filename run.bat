@echo off
REM Launches Nightjar. Keep this window open while you use it; Ctrl+C quits.
cd /d "%~dp0"
".venv\Scripts\python.exe" nightjar.py %*
pause
