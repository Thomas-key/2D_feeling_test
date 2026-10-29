@echo off
REM run_oracle.bat -- is the day's decision learning ruined by a BAD TEACHER?
REM
REM The day accumulates the decision LoRA while the world model stays FROZEN
REM (--per-task-reset world). That scored 79.4% on the held-out set vs 92.2% baseline.
REM
REM Here the LEARNING signal is switched to true physics (--oracle-teacher),
REM while accept/reject still uses the world model -> behaviour unchanged,
REM only the teacher changes.
REM
REM   dec-only, world teacher (already have) : runs/heldout_dec.json   79.4%
REM   dec-only, ORACLE teacher (this run)    : runs/heldout_oracle.json   ?
REM
REM Also compares the world-accuracy probe of the two runs.

setlocal
cd /d "%~dp0.."

set PY=python
set DEV=--device cuda
set HELD=--tasks-file tasks/seed2.json tasks/seed3.json

echo.
echo ==========================================================
echo  STEP 1/3  day on seed1: decision accumulates,
echo            world frozen, teacher = TRUE physics
echo ==========================================================
%PY% train_day.py %DEV% --tasks-file tasks/seed1.json --per-task-reset world --oracle-teacher --out runs/s1_oracle.json --ckpt-out runs/s1_oracle.pt
if errorlevel 1 goto fail

echo.
echo ==========================================================
echo  STEP 2/3  frozen exam on held-out
echo ==========================================================
%PY% train_day.py %DEV% %HELD% --eval-only --base runs/s1_oracle.pt --load-dec-lora --no-save --out runs/heldout_oracle.json
if errorlevel 1 goto fail

echo.
echo ---- vs baseline (no training)
%PY% tool/compare_eval.py runs/heldout_base.json runs/heldout_oracle.json

echo.
echo ---- vs dec-only + world teacher  (the 79.4% collapse)
%PY% tool/compare_eval.py runs/heldout_dec.json runs/heldout_oracle.json

echo.
echo ==========================================================
echo  STEP 3/3  done
echo ==========================================================
echo If ORACLE teacher is fine (around 92%%) -> the world model was the bottleneck.
echo If ORACLE teacher ALSO collapses (around 80%%) -> the learning rule is the problem,
echo   not the teacher, and improving the world model will not fix it.
goto end

:fail
echo.
echo !!!! a step returned an error -- stopping. Send this window back.

:end
endlocal
