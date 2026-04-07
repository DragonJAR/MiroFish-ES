@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion

echo [MiroFish] Deteniendo MiroFish...

REM Matar solo procesos Vite específicos (no todos los node.exe)
wmic process where "name='node.exe' and commandline like '%%vite%%'" delete >nul 2>&1

REM Detectar Docker Compose versión
docker compose version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "DCOMPOSE=docker compose"
) else (
    docker-compose --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "DCOMPOSE=docker-compose"
    ) else (
        echo [WARN] No se encontró docker compose ni docker-compose, saltando detención de Docker
        goto :end
    )
)

if exist docker\graphiti\docker-compose.yml (
    %DCOMPOSE% -f docker\graphiti\docker-compose.yml down >nul 2>&1
)

:end
echo [MiroFish] Detenido
pause
