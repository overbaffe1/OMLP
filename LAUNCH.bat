@echo off
chcp 65001 >nul
rem LAUNCH.bat — start with a clean machine in one click.
rem Details: RUNBOOK.md (full procedure) and README.md (commands).

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m venv .venv
) else (
  python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo === 1) Diagnostics: is the environment alive (82 tests, ~25 sec) ===
python -m pytest tests -q

if "%~1"=="" (
  echo.
  echo Dataset not passed. Usage: LAUNCH.bat "C:\path\to\test"
  echo Inside the folder there should be af/, bs/, meta.csv, sample_submission.csv
  echo Details: RUNBOOK.md
) else (
  echo.
  echo === 2) Dataset diagnostics: what was found in "%~1" ===
  python -m fires.chips --data-dir "%~1"
  echo.
  echo === 3) Full run: inference on "%~1" (~35 sec on 2 cores) ===
  python inference.py --data-dir "%~1" --output submission_fresh.csv --validate
  echo.
  echo Done: submission_fresh.csv is nearby.
  echo Where submission.csv appears is specified in README.md, section 1.
)
echo.
pause
