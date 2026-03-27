#!/bin/bash
# ============================================
# MiroFish - Script de Detención (Linux/Mac)
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[MiroFish] Deteniendo MiroFish..."

# Matar procesos locales
pkill -f "uvicorn.*run:app" 2>/dev/null || true
pkill -f "vite.*--host" 2>/dev/null || true

# Detener Neo4j (si está corriendo)
docker compose -f docker/graphiti/docker-compose.yml down 2>/dev/null || true

echo "[MiroFish] Detenido"
