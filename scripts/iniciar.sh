#!/bin/bash
# ============================================
# MiroFish - Script de Inicio (Linux/Mac)
# ============================================
# Uso: ./scripts/iniciar.sh
# Detiene con: Ctrl+C o ./scripts/detener.sh
# ============================================

set -e

# Detectar project root (scripts/ está en la raíz)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[MiroFish]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# === VERIFICACIONES ===
if [ ! -f .env ]; then
    error ".env no encontrado. Copia .env.example a .env y configura las variables."
    exit 1
fi

# Leer MEMORY_BACKEND y NEO4J_PASSWORD del .env
MEMORY_BACKEND=$(grep -E '^MEMORY_BACKEND=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
NEO4J_PASSWORD=$(grep -E '^NEO4J_PASSWORD=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)

MEMORY_BACKEND="${MEMORY_BACKEND:-zep}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# === CONFIGURACIÓN DE VARIABLES DE ENTORNO PARA GRAPHITI ===
# Cargar variables desde .env
export $(grep -v '^#' .env | grep -v '^$' | xargs)

if [ "$MEMORY_BACKEND" = "graphiti" ]; then
    info "Configurando variables de entorno para Graphiti..."
    
    # Graphiti usa OPENAI_* para el LLM (lee desde LLM_* del proyecto)
    export OPENAI_API_KEY="${LLM_API_KEY}"
    export OPENAI_BASE_URL="${LLM_BASE_URL}"
    export OPENAI_MODEL_NAME="${LLM_MODEL_NAME}"
    
    info "  OPENAI_API_KEY=****"
    info "  OPENAI_BASE_URL=${LLM_BASE_URL}"
    info "  OPENAI_MODEL_NAME=${LLM_MODEL_NAME}"
else
    info "Zep Cloud seleccionado - sin variables OPENAI_* adicionales"
fi

# === LIMPIEZA DE PROCESOS ANTERIORES ===
log "Deteniendo procesos anteriores..."
docker compose --profile graphiti down >/dev/null 2>&1 || true
# Matar procesos npm/uvicorn que puedan quedar
pkill -f "uvicorn.*run:app" 2>/dev/null || true
pkill -f "vite.*--host" 2>/dev/null || true
sleep 1

# === BACKEND SELECTION ===
info "Memory backend: $MEMORY_BACKEND"

NEO4J_STARTED=false

if [ "$MEMORY_BACKEND" = "graphiti" ]; then
    # Verificar Docker disponible
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker no está instalado. Necesario para Neo4j con Graphiti."
        exit 1
    fi

    log "Iniciando Neo4j (Graphiti mode)..."

    # Crear directorios de datos
    mkdir -p backend/neo4j/data backend/neo4j/logs

    # Verificar si ya está corriendo
    if docker ps --format '{{.Names}}' | grep -q "^mirofish-neo4j$"; then
        warn "Neo4j ya está corriendo"
    else
        docker compose -f docker/graphiti/docker-compose.yml up -d neo4j

        log "Esperando Neo4j (max 60s)..."
        waited=0
        while [ $waited -lt 60 ]; do
            if docker exec mirofish-neo4j cypher-shell \
                -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1" >/dev/null 2>&1; then
                break
            fi
            sleep 2
            waited=$((waited + 2))
            printf "."
        done
        echo ""

        if [ $waited -ge 60 ]; then
            error "Neo4j no respondió en 60s. Verifica logs: docker logs mirofish-neo4j"
            docker compose -f docker/graphiti/docker-compose.yml down
            exit 1
        fi

        log "Neo4j listo (http://localhost:7474)"
        NEO4J_STARTED=true
    fi
else
    info "Zep Cloud seleccionado - sin Neo4j local"
fi

# === INICIAR MIROFISH LOCAL ===
log "Iniciando MiroFish (local)..."

cleanup() {
    echo ""
    log "Deteniendo MiroFish..."

    # Matar procesos hijos
    pkill -f "uvicorn.*run:app" 2>/dev/null || true
    pkill -f "vite.*--host" 2>/dev/null || true

    # Detener Neo4j si lo iniciamos nosotros
    if [ "$NEO4J_STARTED" = true ]; then
        log "Deteniendo Neo4j..."
        docker compose -f docker/graphiti/docker-compose.yml down
    fi

    log "Listo!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Iniciar con npm run dev (backend + frontend)
npm run dev &
APP_PID=$!

# Esperar un poco y verificar que arrancó
sleep 3
if ! kill -0 $APP_PID 2>/dev/null; then
    error "MiroFish falló al iniciar. Verifica npm run dev"
    [ "$NEO4J_STARTED" = true ] && docker compose --profile graphiti down
    exit 1
fi

# === MOSTRAR INFO ===
echo ""
echo "=========================================="
echo -e "  ${GREEN}MiroFish iniciado exitosamente${NC}"
echo "=========================================="
echo ""
echo -e "  Frontend:  ${BLUE}http://localhost:3000${NC}"
echo -e "  Backend:   ${BLUE}http://localhost:5001${NC}"
echo -e "  Memory:    ${BLUE}$MEMORY_BACKEND${NC}"
if [ "$MEMORY_BACKEND" = "graphiti" ]; then
    echo -e "  Neo4j UI:  ${BLUE}http://localhost:7474${NC}"
fi
echo ""
echo -e "  Para detener: ${YELLOW}Ctrl+C${NC} o ${YELLOW}./scripts/detener.sh${NC}"
echo ""

# Mantener vivo hasta Ctrl+C
wait $APP_PID
