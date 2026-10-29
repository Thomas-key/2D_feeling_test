@echo off
REM run_worldfast.bat -- does a FASTER-LEARNING world model fix the day?
REM
REM What we know:
REM   decision accumulates + world frozen   + world-model teacher  -> 79.4% (held-out)
REM   decision accumulates + world frozen   + TRUE-physics teacher -> 95.6%  (+29, p<1e-4)
REM   so the teacher's accuracy decides whether the day helps or hurts.
REM
REM But a normal day only moves world accuracy by ~2% (check_world:
REM position 0.2145 -> 0.2076 m). That is nowhere near enough.
REM
REM This run turns the world's learning rate up 100x (--world-lr 1e-2,
REM matching the 95.4% per-task setup) and asks:
REM   1. does the world actually get more accurate?  (check_world)
REM   2. does the held-out set improve?              (compare_eval)

setlocal
cd /d "%~dp0.."

set PY=python
set DEV=--device cuda
set HELD=--tasks-file tasks/seed2.json tasks/seed3.json

echo.
echo ==========================================================
echo  STEP 1/4  day on seed1, both LoRA accumulate,
echo            world learning rate 1e-2 (100x default)
echo ==========================================================
%PY% train_day.py %DEV% --tasks-file tasks/seed1.json --world-lr 1e-2 --out runs/s1_fast.json --ckpt-out runs/s1_fast.pt
if errorlevel 1 goto fail

echo.
echo ==========================================================
echo  STEP 2/4  frozen exam on held-out
echo ==========================================================
%PY% train_day.py %DEV% %HELD% --eval-only --base runs/s1_fast.pt --load-dec-lora --no-save --out runs/heldout_fast.json
if errorlevel 1 goto fail

echo.
echo ---- vs baseline (no training)
%PY% tool/compare_eval.py runs/heldout_base.json runs/heldout_fast.json

echo.
echo ==========================================================
echo  STEP 3/4  how accurate is the world now?
echo            base  /  normal world-lr  /  fast world-lr
echo ==========================================================
%PY% tool/check_world.py --device cuda --samples 2000 --ckpts pretrain_base.pt runs/s1_both.pt runs/s1_fast.pt
if errorlevel 1 goto fail

echo.
echo ==========================================================
echo  STEP 4/4  done
echo ==========================================================
echo Read it like this:
echo   [a] world position error drops a lot AND held-out goes up
echo       => fix the world model, the day follows
echo   [b] world error drops a lot but held-out stays near 92
echo       => accuracy alone is not enough
echo   [c] world error gets WORSE (pain / position)
echo       => 1e-2 is too big, tune it down
goto end

:fail
echo.
echo !!!! a step returned an error -- stopping. Send this window back.

:end
endlocal
