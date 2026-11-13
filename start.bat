@echo off
call conda activate Pitcher
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0Src\UI_Control"
python main.py
pause
