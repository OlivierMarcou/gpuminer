@echo off
echo ================================================
echo VERIFICATION DES FICHIERS CORRIGES
echo ================================================
echo.

echo Verification cuda_miner.cu...
findstr /C:"char username[256]" cuda_miner.cu >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] cuda_miner.cu contient "char username[256]"
) else (
    echo [ERREUR] cuda_miner.cu NE contient PAS "char username[256]"
    echo MAUVAIS FICHIER! Recommence la copie!
    pause
    exit /b 1
)

echo.
echo Verification kawpow.cu...
findstr /C:"g_header_hash[i*4+0]" kawpow.cu >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] kawpow.cu contient l'acces memoire corrige
) else (
    echo [ERREUR] kawpow.cu NE contient PAS l'acces memoire corrige
    echo MAUVAIS FICHIER! Recommence la copie!
    pause
    exit /b 1
)

echo.
echo ================================================
echo TOUS LES FICHIERS SONT CORRECTS!
echo Tu peux compiler maintenant!
echo ================================================
echo.
echo Lance: build_simple.bat
echo.
pause
