@echo off
echo ========================================
echo TEST COMPILATION ETHASH OPTIMISE
echo ========================================
echo.

REM Nettoyer
del *.obj 2>nul
del *.exe 2>nul

echo Test 1: Compilation ethash.cu...
nvcc -c ethash.cu -o ethash.obj
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] ethash.cu ne compile pas!
    pause
    exit /b 1
)
echo [OK] ethash.cu compile

echo.
echo Test 2: Compilation cuda_miner.cu...
nvcc -c cuda_miner.cu -o cuda_miner.obj
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] cuda_miner.cu ne compile pas!
    pause
    exit /b 1
)
echo [OK] cuda_miner.cu compile

echo.
echo Test 3: Compilation stratum.c...
cl /c stratum.c
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] stratum.c ne compile pas!
    pause
    exit /b 1
)
echo [OK] stratum.c compile

echo.
echo Test 4: Compilation cJSON.c...
cl /c cJSON.c
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] cJSON.c ne compile pas!
    pause
    exit /b 1
)
echo [OK] cJSON.c compile

echo.
echo Test 5: Linking...
nvcc -o cuda_miner.exe cuda_miner.obj ethash.obj stratum.obj cJSON.obj -lws2_32
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Linking a echoue!
    pause
    exit /b 1
)
echo [OK] Linking reussi

echo.
echo ========================================
echo TOUS LES TESTS PASSES !
echo ========================================
echo cuda_miner.exe est pret a etre utilise
echo.
pause
