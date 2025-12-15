@echo off
REM Script de compilation pour CryptoMiner CUDA - Windows

echo ==========================================
echo CryptoMiner CUDA - Compilation
echo ==========================================
echo.

REM Vérifier nvcc (CUDA Toolkit)
echo [1/4] Verification de CUDA Toolkit...
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERREUR: nvcc non trouve!
    echo.
    echo Installez CUDA Toolkit depuis:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    echo Puis ajoutez au PATH:
    echo   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin
    echo.
    pause
    exit /b 1
)

nvcc --version
echo OK: CUDA Toolkit detecte
echo.

REM Vérifier Visual Studio (cl.exe)
echo [2/4] Verification de Visual Studio...
where cl >nul 2>&1
REM Nettoyer les anciens fichiers
echo [3/4] Nettoyage...
if exist *.obj del /Q *.obj
if exist *.exp del /Q *.exp
if exist *.lib del /Q *.lib
echo OK
echo.

REM Compilation
echo [4/4] Compilation...
echo.

echo Compilation SHA256 kernel...
nvcc -O3 -arch=sm_50 ^
     -gencode=arch=compute_50,code=sm_50 ^
     -gencode=arch=compute_60,code=sm_60 ^
     -gencode=arch=compute_70,code=sm_70 ^
     -gencode=arch=compute_75,code=sm_75 ^
     -gencode=arch=compute_80,code=sm_80 ^
     -gencode=arch=compute_86,code=sm_86 ^
     -gencode=arch=compute_89,code=sm_89 ^
     --ptxas-options=-v ^
     -c sha256.cu -o sha256.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation SHA256!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation Ethash kernel avec DAG...
nvcc -O3 -arch=sm_50 ^
     -gencode=arch=compute_50,code=sm_50 ^
     -gencode=arch=compute_60,code=sm_60 ^
     -gencode=arch=compute_70,code=sm_70 ^
     -gencode=arch=compute_75,code=sm_75 ^
     -gencode=arch=compute_80,code=sm_80 ^
     -gencode=arch=compute_86,code=sm_86 ^
     -gencode=arch=compute_89,code=sm_89 ^
     --ptxas-options=-v ^
     -c ethash.cu -o ethash.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation Ethash!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation Equihash kernel...
nvcc -O3 -arch=sm_50 ^
     -gencode=arch=compute_50,code=sm_50 ^
     -gencode=arch=compute_60,code=sm_60 ^
     -gencode=arch=compute_70,code=sm_70 ^
     -gencode=arch=compute_75,code=sm_75 ^
     -gencode=arch=compute_80,code=sm_80 ^
     -gencode=arch=compute_86,code=sm_86 ^
     -gencode=arch=compute_89,code=sm_89 ^
     --ptxas-options=-v ^
     -c equihash.cu -o equihash.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation Equihash!
    pause
    exit /b 1
)
echo OK
echo.

echo Compilation Equihash 192,7 kernel (Zcash)...
nvcc -O3 -arch=sm_50 ^
     -gencode=arch=compute_50,code=sm_50 ^
     -gencode=arch=compute_60,code=sm_60 ^
     -gencode=arch=compute_70,code=sm_70 ^
     -gencode=arch=compute_75,code=sm_75 ^
     -gencode=arch=compute_80,code=sm_80 ^
     -gencode=arch=compute_86,code=sm_86 ^
     -gencode=arch=compute_89,code=sm_89 ^
     -c equihash_192_7.cu -o equihash_192_7.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation Equihash 192,7!
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
     -gencode=arch=compute_80,code=sm_80 ^
     -gencode=arch=compute_86,code=sm_86 ^
     -gencode=arch=compute_89,code=sm_89 ^
     --ptxas-options=-v ^
     -c cuda_miner.cu -o cuda_miner.obj

if %ERRORLEVEL% NEQ 0 (
    echo ERREUR compilation programme principal!
    pause
    exit /b 1
)
echo OK
echo.

echo Linkage final...
nvcc -O3 -o cuda_miner.exe cuda_miner.obj sha256.obj ethash.obj equihash.obj equihash_192_7.obj stratum.obj cJSON.obj -lws2_32

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
echo Executable cree: cuda_miner.exe
echo.
echo Pour lancer:
echo   cuda_miner.exe
echo.

REM Afficher infos sur le binaire
if exist cuda_miner.exe (
    echo Taille du fichier:
    dir cuda_miner.exe | find "cuda_miner.exe"
    echo.
)

echo Voulez-vous lancer le mineur maintenant? (O/N)
set /p LAUNCH=
if /i "%LAUNCH%"=="O" (
    echo.
    echo Lancement...
    echo.
    cuda_miner.exe
)

pause
