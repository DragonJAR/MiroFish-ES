@echo off
chcp 65001 >nul 2>nul
setlocal enabledelayedexpansion

echo ============================================
echo   MiroFish - Iniciando (Windows)
echo ============================================
echo.

REM === VERIFICAR .ENV ===
if not exist ".env" (
    echo [ERROR] .env no encontrado
    echo Copia .env.example a .env y configura las variables
    pause
    exit /b 1
)

REM Leer variables del .env
for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
    if not "%%b"=="" (
        set "line=%%b"
        set "line=!line:"=!"
        set "%%a=!line!"
    )
)

if not defined MEMORY_BACKEND set MEMORY_BACKEND=zep
if not defined NEO4J_PASSWORD set NEO4J_PASSWORD=mirofish.dragonjar

REM === DETENER ANTERIOR ===
echo [MiroFish] Deteniendo procesos anteriores...
if exist docker\graphiti\docker-compose.yml (
    docker compose -f docker\graphiti\docker-compose.yml down >nul 2>&1
)
REM Matar procesos específicos de MiroFish (Vite)
wmic process where "name='node.exe' and commandline like '%%vite%%'" delete >nul 2>&1

REM === BACKEND SELECTION ===
echo [INFO] Memory backend: %MEMORY_BACKEND%

REM Detectar Docker Compose versión
docker --version >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Docker no esta instalado.
    pause
    exit /b 1
)

REM Probar docker compose (v2) primero
docker compose version >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "DCOMPOSE=docker compose"
) else (
    REM Fallback a docker-compose (v1)
    docker-compose --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "DCOMPOSE=docker-compose"
    ) else (
        echo [ERROR] Ni docker compose ni docker-compose encontrados.
        pause
        exit /b 1
    )
)

if "%MEMORY_BACKEND%"=="graphiti" (
    echo [MiroFish] Iniciando Neo4j (Graphiti mode)...
    %DCOMPOSE% -f docker\graphiti\docker-compose.yml up -d neo4j

    echo [MiroFish] Esperando Neo4j (max 60s)...
    set /a waited=0
    :wait_neo4j
    docker exec mirofish-neo4j cypher-shell -u neo4j -p "%NEO4J_PASSWORD%" "RETURN 1" >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        timeout /t 2 /nobreak >nul
        set /a waited+=2
        if !waited! lss 60 (
            <nul set /p "=."
            goto wait_neo4j
        )
        echo.
        echo [ERROR] Neo4j no respondio en 60s
        %DCOMPOSE% -f docker\graphiti\docker-compose.yml down
        pause
        exit /b 1
    )
    echo.
    echo [MiroFish] Neo4j listo en http://localhost:7474
) else (
    echo [INFO] Zep Cloud seleccionado
)

REM === INICIAR MIROFISH LOCAL ===
echo [MiroFish] Iniciando MiroFish (local)...
echo.
echo ============================================
echo   MiroFish iniciado!
echo ============================================
echo.
echo   Frontend:  http://localhost:3000
echo   Backend:   http://localhost:5001
echo   Memory:    %MEMORY_BACKEND%
if "%MEMORY_BACKEND%"=="graphiti" (
    echo   Neo4j UI:  http://localhost:7474
)
echo.
echo   Para detener: Ctrl+C o scripts\detener.bat
echo.

npm run dev

REM === LIMPIEZA ===
echo.
echo [MiroFish] Deteniendo...
if exist docker\graphiti\docker-compose.yml (
    %DCOMPOSE% -f docker\graphiti\docker-compose.yml down >nul 2>&1
)
echo [MiroFish] Detenido
