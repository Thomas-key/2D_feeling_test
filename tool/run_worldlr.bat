@echo off
REM run_worldlr.bat -- where does the world learning rate stop being useful?
REM
REM We already have two points:
REM   --world-lr 1e-4 (default) : load error 0.028 -> 0.023 (better), held-out 92.8%
REM   --world-lr 1e-2 (100x)    : load error 0.028 -> 0.111 (4x worse), held-out 44.4%
REM
REM Same update rule: small steps help, big steps destroy. This sweep fills the gap
REM with 3e-4 and 1e-3 so we can see where it turns.
REM
REM Each run also prints the failure mix (stall vs unsafe). The 1e-2 disaster had
REM a signature: during the day 27/38 failures were "unsafe" (safety gate blocked
REM by a pessimistic world), and after freezing 99/100 were "stall" (the decision
REM wandered instead of approaching).

setlocal
cd /d "%~dp0.."

set PY=python
set DEV=--device cuda
set HELD=--tasks-file tasks/seed2.json tasks/seed3.json

for %%L in (3e-4 1e-3) do call :one %%L
goto summary

:one
echo.
echo ==========================================================
echo  world-lr %1 : day on seed1
echo ==========================================================
%PY% train_day.py %DEV% --tasks-file tasks/seed1.json --world-lr %1 --out runs/s1_w%1.json --ckpt-out runs/s1_w%1.pt
if errorlevel 1 goto fail

echo.
echo ---- frozen exam on held-out
%PY% train_day.py %DEV% %HELD% --eval-only --base runs/s1_w%1.pt --load-dec-lora --no-save --out runs/heldout_w%1.json
if errorlevel 1 goto fail

%PY% tool/compare_eval.py runs/heldout_base.json runs/heldout_w%1.json

echo.
echo ---- failure mix after freezing
%PY% tool/show_fails.py runs/heldout_w%1.json --summary
echo ---- failure mix during the day
%PY% tool/show_fails.py runs/s1_w%1.json --summary
exit /b 0

:summary
echo.
echo ==========================================================
echo  world accuracy at each learning rate
echo ==========================================================
%PY% tool/check_world.py --device cuda --samples 2000 --ckpts pretrain_base.pt runs/s1_both.pt runs/s1_w3e-4.pt runs/s1_w1e-3.pt runs/s1_fast.pt
echo.
echo Read it like this:
echo   [a] error drops more as lr grows, held-out also drops
echo       => pick the lr just before held-out starts falling
echo   [b] error drops but held-out is flat
echo       => the world is getting more accurate but the decision cannot use it
echo   [c] some lr makes load/pain error jump while held-out collapses
echo       => that is the pessimistic blow-up; stay well below it
goto end

:fail
echo.
echo !!!! a step returned an error -- stopping. Send this window back.

:end
endlocal
