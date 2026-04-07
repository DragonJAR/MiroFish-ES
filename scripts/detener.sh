#!/bin/bash
# ============================================
# MiroFish - Script de Detención (Linux/Mac)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[MiroFish] Deteniendo MiroFish..."

# Detener backend usando PID guardado
if [ -f /tmp/mirofish_backend.pid ]; then
    BACKEND_PID=$(cat /tmp/mirofish_backend.pid)
    if [ -n "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "[MiroFish] Backend detenido (PID: $BACKEND_PID)"
    else
        # Fallback: intentar con pkill específico (no usará patrones muy amplios)
        pkill -f "uvicorn.*run:app" 2>/dev/null || true
    fi
    rm -f /tmp/mirofish_backend.pid
else
    # Fallback si no existe archivo PID
    pkill -f "uvicorn.*run:app" 2>/dev/null || true
fi

# Detener frontend usando PID guardado
if [ -f /tmp/mirofish_frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/mirofish_frontend.pid)
    if [ -n "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "[MiroFish] Frontend detenido (PID: $FRONTEND_PID)"
    else
        # Fallback: intentar con pkill específico
        pkill -f "vite.*--host.*3000" 2>/dev/null || true
    fi
    rm -f /tmp/mirofish_frontend.pid
else
    # Fallback si no existe archivo PID
    pkill -f "vite.*--host.*3000" 2>/dev/null || true
fi

# Detener Neo4j (si está corriendo)
docker compose -f docker/graphiti/docker-compose.yml down 2>/dev/null || true

echo "[MiroFish] Detenido"
