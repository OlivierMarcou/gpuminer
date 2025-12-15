@echo off
REM Script de compilation SIMPLIFIE - Ethash seulement
echo ==========================================
echo Compilation ETHASH MINER
echo ==========================================
echo.

REM Nettoyer
echo Nettoyage...
if exist *.obj del /Q *.obj
if exist *.exe del /Q *.exe
echo OK
echo.

echo Compilation Ethash kernel...
nvcc -O3 -arch=sm_50 ^
     -gencode=arch=compute_50,code=sm_50 ^
     -gencode=arch=compute_60,code=sm_60 ^
     -gencode=arch=compute_70,code=sm_70 ^
     -gencode=arch=compute_75,code=sm_75 ^
     -c ethash.cu -o ethash.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation Ethash!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation Stratum client...
nvcc -O3 -c stratum.c -o stratum.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation Stratum!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation cJSON...
nvcc -O3 -c cJSON.c -o cJSON.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation cJSON!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation programme principal...
nvcc -O3 -arch=sm_50 ^
     -gencode=arch=compute_50,code=sm_50 ^
     -gencode=arch=compute_60,code=sm_60 ^
     -gencode=arch=compute_70,code=sm_70 ^
     -gencode=arch=compute_75,code=sm_75 ^
     -c cuda_miner.cu -o cuda_miner.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation programme principal!
    pause
    exit /b 1
)
echo OK
echo.

echo Linkage final...
nvcc -O3 -o cuda_miner.exe cuda_miner.obj ethash.obj stratum.obj cJSON.obj -lws2_32

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR linkage!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo COMPILATION REUSSIE!
echo ==========================================
echo.
echo Executable: cuda_miner.exe
echo.
if exist cuda_miner.exe (
    dir cuda_miner.exe | find "cuda_miner.exe"
)
echo.
pause
