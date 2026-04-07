#!/bin/bash
# ============================================
# MiroFish - Script de Inicio (Linux/Mac)
# ============================================
# Uso: ./scripts/iniciar.sh
# Detiene con: Ctrl+C o ./scripts/detener.sh
# ============================================

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

# Leer config del .env (sin exportar todavía)
MEMORY_BACKEND=$(grep -E '^MEMORY_BACKEND=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
NEO4J_PASSWORD=$(grep -E '^NEO4J_PASSWORD=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
LLM_API_KEY=$(grep -E '^LLM_API_KEY=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
LLM_BASE_URL=$(grep -E '^LLM_BASE_URL=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
LLM_MODEL_NAME=$(grep -E '^LLM_MODEL_NAME=' .env 2>/dev/null | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)

MEMORY_BACKEND="${MEMORY_BACKEND:-zep}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

if [ -z "$LLM_API_KEY" ]; then
    error "LLM_API_KEY no está configurada en .env"
    exit 1
fi

# === EXPORTAR VARIABLES PARA PROCESOS HIJOS ===
# Exportar TODO el .env al entorno (necesario para Graphiti OPENAI_*)
set -a
while IFS='=' read -r key value; do
    # Solo exportar si tiene key y no es comentario/vacío
    if [[ -n "$key" && ! "$key" =~ ^# && ! "$key" =~ ^[[:space:]] ]]; then
        # Limpiar value de comillas
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        export "$key=$value"
    fi
done < <(grep -v '^#' .env | grep -v '^$' | grep -v '^ ')
set +a

# Para Graphiti, mapear LLM_* a OPENAI_* (Graphiti usa la SDK de OpenAI internamente)
if [ "$MEMORY_BACKEND" = "graphiti" ]; then
    export OPENAI_API_KEY="${LLM_API_KEY}"
    export OPENAI_BASE_URL="${LLM_BASE_URL}"
    export OPENAI_MODEL_NAME="${LLM_MODEL_NAME}"
    info "Configurando variables de entorno para Graphiti..."
    info "  OPENAI_API_KEY=****"
    info "  OPENAI_BASE_URL=${LLM_BASE_URL}"
    info "  OPENAI_MODEL_NAME=${LLM_MODEL_NAME}"
else
    info "Zep Cloud seleccionado"
fi

info "Memory backend: $MEMORY_BACKEND"

# === LIMPIEZA DE PUERTOS ===
log "Limpiando procesos anteriores..."

kill_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pids" ]; then
        warn "Puerto $port ocupado — matando proceso(es)..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

kill_port 5001  # Backend Flask
kill_port 3000  # Frontend Vite

log "Puertos limpios"

# === NEO4J (solo Graphiti) ===
NEO4J_STARTED=false

if [ "$MEMORY_BACKEND" = "graphiti" ]; then
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker no está instalado. Necesario para Neo4j con Graphiti."
        exit 1
    fi

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^mirofish-neo4j$"; then
        warn "Neo4j ya está corriendo"
    else
        log "Iniciando Neo4j..."
        mkdir -p backend/neo4j/data backend/neo4j/logs
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
            error "Neo4j no respondió en 60s. Verifica: docker logs mirofish-neo4j"
            docker compose -f docker/graphiti/docker-compose.yml down
            exit 1
        fi

        log "Neo4j listo (http://localhost:7474)"
        NEO4J_STARTED=true
    fi
fi

# === INICIAR MIROFISH ===
log "Iniciando MiroFish..."

cleanup() {
    echo ""
    log "Deteniendo MiroFish..."
    kill_port 5001
    kill_port 3000
    if [ "$NEO4J_STARTED" = true ]; then
        log "Deteniendo Neo4j..."
        docker compose -f docker/graphiti/docker-compose.yml down
    fi
    log "Listo!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Lanzar backend y frontend directamente (no via npm para preservar env vars)
cd backend && uv run python run.py > "$PROJECT_ROOT/backend/logs/app.log" 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

cd frontend && npx vite --host 0.0.0.0 --port 3000 > /dev/null 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Esperar y verificar que levantaron
sleep 5

BACKEND_OK=false
FRONTEND_OK=false

if kill -0 $BACKEND_PID 2>/dev/null && lsof -i :5001 >/dev/null 2>&1; then
    BACKEND_OK=true
fi

if kill -0 $FRONTEND_PID 2>/dev/null && lsof -i :3000 >/dev/null 2>&1; then
    FRONTEND_OK=true
fi

if [ "$BACKEND_OK" = false ] && [ "$FRONTEND_OK" = false ]; then
    error "MiroFish no pudo iniciar. Revisá los logs:"
    echo "  tail -20 backend/logs/app.log"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    [ "$NEO4J_STARTED" = true ] && docker compose -f docker/graphiti/docker-compose.yml down
    exit 1
fi

# === MOSTRAR INFO ===
echo ""
echo "=========================================="
echo -e "  ${GREEN}MiroFish iniciado${NC}"
echo "=========================================="
echo ""
[ "$BACKEND_OK" = true ]  && echo -e "  Backend:   ${GREEN}✓${NC} ${BLUE}http://localhost:5001${NC}" || echo -e "  Backend:   ${RED}✗${NC} puerto 5001"
[ "$FRONTEND_OK" = true ] && echo -e "  Frontend:  ${GREEN}✓${NC} ${BLUE}http://localhost:3000${NC}" || echo -e "  Frontend:  ${RED}✗${NC} puerto 3000"
echo -e "  Memory:    ${BLUE}$MEMORY_BACKEND${NC}"
[ "$MEMORY_BACKEND" = "graphiti" ] && echo -e "  Neo4j UI:  ${BLUE}http://localhost:7474${NC}"
echo ""
echo -e "  Logs:      ${BLUE}backend/logs/app.log${NC}"
echo -e "  Para detener: ${YELLOW}Ctrl+C${NC} o ${YELLOW}./scripts/detener.sh${NC}"
echo ""

# Mantener vivo hasta Ctrl+C
wait $BACKEND_PID $FRONTEND_PID
