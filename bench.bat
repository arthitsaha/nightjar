@echo off
REM Records 6 seconds from the console and times each pipeline stage.
cd /d "%~dp0"
".venv\Scripts\python.exe" nightjar.py --bench
pause
