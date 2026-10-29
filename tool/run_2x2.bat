@echo off
REM run_2x2.bat -- day-learning 2x2 on a HELD-OUT set (seed2+seed3)
REM
REM Day is trained ONLY on seed1 (87 tasks). The exam is seed2+seed3 (180 tasks
REM never used in training). Same-task exams measure fitting, not learning.
REM
REM Results are also written to runs\*.json (compare_eval needs them),
REM but everything you need is printed to this window.
REM
REM Usage:  double-click, or from the repo root:  tool\run_2x2.bat

setlocal
cd /d "%~dp0.."

set PY=python
set DEV=--device cuda
set DAY=--tasks-file tasks/seed1.json
set HELD=--tasks-file tasks/seed2.json tasks/seed3.json

echo.
echo ==========================================================
echo  STEP 0/7  baseline: no training, frozen, on held-out set
echo ==========================================================
%PY% train_day.py %DEV% %HELD% --eval-only --base pretrain_base.pt --out runs/heldout_base.json
if errorlevel 1 goto fail

echo.
echo ==========================================================
echo  STEP 1/7  day on seed1   [1] decision + world accumulate
echo ==========================================================
%PY% train_day.py %DEV% %DAY% --out runs/s1_both.json --ckpt-out runs/s1_both.pt
if errorlevel 1 goto fail

echo.
echo ==========================================================
echo  STEP 2/7  frozen exam on held-out   [1]
echo ==========================================================
%PY% train_day.py %DEV% %HELD% --eval-only --base runs/s1_both.pt --load-dec-lora --no-save --out runs/heldout_both.json
if errorlevel 1 goto fail

echo.
echo ---- compare: baseline  vs  [1] decision+world accumulate
%PY% tool/compare_eval.py runs/heldout_base.json runs/heldout_both.json

echo.
echo ==========================================================
echo  STEP 3/7  day on seed1   [2] ONLY world accumulates
echo            (decision LoRA reset every task)
echo ==========================================================
%PY% train_day.py %DEV% %DAY% --per-task-reset dec --out runs/s1_world.json --ckpt-out runs/s1_world.pt
if errorlevel 1 goto fail

echo.
echo ==========================================================
echo  STEP 4/7  frozen exam on held-out   [2]
echo ==========================================================
%PY% train_day.py %DEV% %HELD% --eval-only --base runs/s1_world.pt --load-dec-lora --no-save --out runs/heldout_world.json
if errorlevel 1 goto fail

echo.
echo ---- compare: baseline  vs  [2] only world accumulates
%PY% tool/compare_eval.py runs/heldout_base.json runs/heldout_world.json

echo.
echo ==========================================================
echo  STEP 5/7  day on seed1   [3] ONLY decision accumulates
echo            (world LoRA reset every task)
echo ==========================================================
%PY% train_day.py %DEV% %DAY% --per-task-reset world --out runs/s1_dec.json --ckpt-out runs/s1_dec.pt
if errorlevel 1 goto fail

echo.
echo ==========================================================
echo  STEP 6/7  frozen exam on held-out   [3]
echo ==========================================================
%PY% train_day.py %DEV% %HELD% --eval-only --base runs/s1_dec.pt --load-dec-lora --no-save --out runs/heldout_dec.json
if errorlevel 1 goto fail

echo.
echo ---- compare: baseline  vs  [3] only decision accumulates
%PY% tool/compare_eval.py runs/heldout_base.json runs/heldout_dec.json

echo.
echo ==========================================================
echo  STEP 7/7  done -- send the whole window back
echo ==========================================================
echo Look at the three "compare" blocks:
echo   [1] both accumulate   [2] world only   [3] decision only
echo Whichever pair shows a NEGATIVE "B net win" means the day made it worse.
echo "p" above 0.05 means the difference is inside noise.
goto end

:fail
echo.
echo !!!! a step returned an error -- stopping. Send this window back.

:end
endlocal
