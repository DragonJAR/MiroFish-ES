@echo off
chcp 65001 >nul 2>nul
echo [MiroFish] Deteniendo MiroFish...

taskkill /F /IM node.exe >nul 2>&1
docker compose --profile graphiti down >nul 2>&1

echo [MiroFish] Detenido
pause
