@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Once README_TR.md dosyasindaki kurulumu tamamlayin.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
