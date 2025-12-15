@echo off
echo ==========================================
echo CryptoMiner CUDA - Compilation
echo ==========================================
echo.

REM Nettoyer
echo [1/3] Nettoyage...
if exist *.obj del /Q *.obj
if exist *.exp del /Q *.exp
if exist *.lib del /Q *.lib
if exist *.exe del /Q *.exe
echo OK
echo.

REM Compilation
echo [2/3] Compilation...
echo.

echo Compilation SHA256 kernel...
nvcc -O3 -arch=sm_50 -c sha256.cu -o sha256.obj
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation SHA256!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation Ethash kernel...
nvcc -O3 -arch=sm_50 -c ethash.cu -o ethash.obj
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation Ethash!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation KawPow kernel...
nvcc -O3 -arch=sm_50 -c kawpow.cu -o kawpow.obj
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation KawPow!
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

echo Compilation config reader...
nvcc -O3 -c config_reader.c -o config_reader.obj
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation config_reader!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation programme principal...
nvcc -O3 -arch=sm_50 -c cuda_miner.cu -o cuda_miner.obj
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation programme principal!
    pause
    exit /b 1
)
echo OK
echo.

echo [3/3] Linkage final...
nvcc -O3 -o cuda_miner.exe cuda_miner.obj sha256.obj ethash.obj kawpow.obj stratum.obj cJSON.obj config_reader.obj -lws2_32
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
echo Executable cree avec succes
if exist cuda_miner.exe (
    dir cuda_miner.exe | find "cuda_miner.exe"
)
echo.
pause
