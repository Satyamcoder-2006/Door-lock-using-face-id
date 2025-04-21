@echo off
echo Clearing access logs...
python clear_logs_cmd.py
if %ERRORLEVEL% EQU 0 (
    echo Logs cleared successfully.
) else (
    echo Failed to clear logs.
)
pause
