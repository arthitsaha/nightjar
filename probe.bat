@echo off
REM Finds out whether your Fn key reaches Windows at all.
cd /d "%~dp0"
".venv\Scripts\python.exe" probe_fn.py
pause
