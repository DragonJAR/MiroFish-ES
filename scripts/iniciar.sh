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

# === DETECCIÓN DE OS ===
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       echo "unknown" ;;
    esac
}

OS_TYPE=$(detect_os)

# === LIMPIEZA DE PUERTOS (CROSS-PLATFORM) ===
log "Limpiando procesos anteriores..."

kill_port() {
    local port=$1
    local pids=""

    case "$OS_TYPE" in
        macos)
            pids=$(lsof -ti :$port 2>/dev/null)
            ;;
        linux)
            pids=$(ss -tlnp 2>/dev/null | grep ":$port " | awk '{print $NF}' | grep -oP 'pid=\K[0-9]+')
            if [ -z "$pids" ]; then
                pids=$(fuser $port/tcp 2>/dev/null)
            fi
            ;;
        *)
            warn "OS no soportado para limpieza de puertos: $OS_TYPE"
            return
            ;;
    esac

    if [ -n "$pids" ]; then
        warn "Puerto $port ocupado — matando proceso(es): $pids"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

kill_port 5001  # Backend Flask
kill_port 3000  # Frontend Vite

log "Puertos limpios"

# === ESPERA DE SERVICIO (HEALTH CHECK) ===
wait_for_service() {
    local name=$1
    local url=$2
    local max_wait=${3:-30}
    local check_interval=${4:-1}

    info "Esperando $name..."
    local waited=0

    while [ $waited -lt $max_wait ]; do
        if curl -sSf "$url" >/dev/null 2>&1; then
            echo ""
            log "$name listo"
            return 0
        fi
        printf "."
        sleep $check_interval
        waited=$((waited + check_interval))
    done

    echo ""
    error "$name no respondió en ${max_wait}s"
    return 1
}

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

        if ! wait_for_service "Neo4j" "http://localhost:7474" 60 2; then
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

# Asegurar directorio de logs existe
mkdir -p "$PROJECT_ROOT/backend/logs"

# PIDs para cleanup
BACKEND_PID=""
FRONTEND_PID=""
BACKEND_LOG_PID=""

cleanup() {
    echo ""
    log "Deteniendo MiroFish..."
    
    # Detener proceso de monitoreo de logs si existe
    if [ -n "$BACKEND_LOG_PID" ] && kill -0 $BACKEND_LOG_PID 2>/dev/null; then
        kill $BACKEND_LOG_PID 2>/dev/null || true
    fi
    
    if [ -n "$BACKEND_PID" ] && kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID 2>/dev/null || true
        info "Backend detenido (PID: $BACKEND_PID)"
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID 2>/dev/null || true
        info "Frontend detenido (PID: $FRONTEND_PID)"
    fi
    if [ "$NEO4J_STARTED" = true ]; then
        log "Deteniendo Neo4j..."
        docker compose -f docker/graphiti/docker-compose.yml down
    fi
    log "Listo!"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Lanzar backend con logs filtrados en vivo
info "Iniciando backend..."

cd backend
# Backend escribe a app.log, monitoreamos ese archivo en background
uv run python run.py > "$PROJECT_ROOT/backend/logs/app.log" 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Iniciar monitoreo de logs filtrados en vivo
(
    LOG_FILE="$PROJECT_ROOT/backend/logs/app.log"
    
    # Esperar a que el archivo exista
    for i in $(seq 1 30); do
        [ -s "$LOG_FILE" ] && break
        ! kill -0 $BACKEND_PID 2>/dev/null && exit 0
        sleep 0.2
    done
    
    # Primero mostrar líneas ya escritas (startup que pasó mientras esperábamos)
    if [ -s "$LOG_FILE" ]; then
        grep -vE '^\d+\.\d+\.\d+\.\d+ .* "(GET|POST|PUT|DELETE) /api/' "$LOG_FILE" | while IFS= read -r line; do
            if echo "$line" | grep -qiE "ERROR|Traceback|Exception|failed|Failed|Address already|refused"; then
                echo -e "${RED}[Backend]${NC} $line"
            elif echo "$line" | grep -qiE "WARNING"; then
                echo -e "${YELLOW}[Backend]${NC} $line"
            else
                echo -e "${BLUE}[Backend]${NC} $line"
            fi
        done
    fi
    
    # Luego hacer tail -f para líneas nuevas (se detiene cuando backend muere)
    tail -f --pid=$BACKEND_PID "$LOG_FILE" 2>/dev/null | grep --line-buffered -vE '^\d+\.\d+\.\d+\.\d+ .* "(GET|POST|PUT|DELETE) /api/' | while IFS= read -r line; do
        if echo "$line" | grep -qiE "ERROR|Traceback|Exception|failed|Failed|Address already|refused"; then
            echo -e "${RED}[Backend]${NC} $line"
        elif echo "$line" | grep -qiE "WARNING"; then
            echo -e "${YELLOW}[Backend]${NC} $line"
        else
            echo -e "${BLUE}[Backend]${NC} $line"
        fi
    done
) &
BACKEND_LOG_PID=$!

# Lanzar frontend (silencioso como antes)
cd frontend && npx vite --host 0.0.0.0 --port 3000 > /dev/null 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Esperar y verificar que levantaron con health checks
BACKEND_OK=false
FRONTEND_OK=false

info "Verificando servicios..."

# Verificar backend (usar /health endpoint, no / que devuelve 404)
if wait_for_service "Backend Flask" "http://localhost:5001/health" 30 1; then
    BACKEND_OK=true
    # Detener monitoreo de logs - backend está UP
    if [ -n "$BACKEND_LOG_PID" ] && kill -0 $BACKEND_LOG_PID 2>/dev/null; then
        kill $BACKEND_LOG_PID 2>/dev/null || true
    fi
else
    # Si backend falló, mostrar las últimas líneas de error
    warn "Backend no respondió. Mostrando últimas líneas del log:"
    if [ -f "$PROJECT_ROOT/backend/logs/app.log" ]; then
        echo -e "${RED}$(tail -20 "$PROJECT_ROOT/backend/logs/app.log" | sed 's/^/  /')${NC}"
    fi
fi

# Verificar frontend
if wait_for_service "Frontend Vite" "http://localhost:3000" 30 1; then
    FRONTEND_OK=true
else
    warn "Frontend no respondió, pero proceso sigue corriendo."
fi

if [ "$BACKEND_OK" = false ] && [ "$FRONTEND_OK" = false ]; then
    error "MiroFish no pudo iniciar correctamente. Revisá los logs:"
    echo "  tail -20 backend/logs/app.log"
    cleanup
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
