#!/bin/bash
# ============================================================================
# sync-upstream.sh - Sincroniza con el repositorio original sin perder cambios locales
#
# USO:
#   ./sync-upstream.sh           # Sincroniza normal
#   ./sync-upstream.sh --dry-run # Solo muestra qué pasaría
#   ./sync-upstream.sh --force   # Resuelve conflictos favoreciendo cambios locales
#
# REQUISITOS:
#   - El remote "upstream" debe apuntar al repo original
# ============================================================================

set -e

DRY_RUN=false
FORCE_LOCAL=false

for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --force) FORCE_LOCAL=true ;;
  esac
done

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MiroFish-ES Sync Upstream ===${NC}"
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "package.json" ] || [ ! -d "backend" ]; then
  echo -e "${RED}Error: Este script debe ejecutarse desde la raíz del proyecto${NC}"
  exit 1
fi

# Verificar remote upstream
if ! git remote | grep -q "^upstream$"; then
  echo -e "${YELLOW}Configurando remote 'upstream'...${NC}"
  git remote add upstream https://github.com/666ghj/MiroFish.git
fi

echo -e "${BLUE}1. Obteniendo últimos cambios del upstream...${NC}"
git fetch upstream

# Mostrar si hay cambios nuevos
UPSTREAM_CHANGES=$(git log HEAD..upstream/main --oneline 2>/dev/null || echo "")
if [ -z "$UPSTREAM_CHANGES" ]; then
  echo -e "${GREEN}✓ El upstream está actualizado, no hay cambios nuevos${NC}"
  exit 0
fi

echo -e "${YELLOW}Cambios disponibles en upstream:${NC}"
echo "$UPSTREAM_CHANGES"
echo ""

if [ "$DRY_RUN" = true ]; then
  echo -e "${BLUE}[DRY-RUN] Se sincronizarían estos cambios${NC}"
  exit 0
fi

# Archivos de traducción que SIEMPRE queremos preservar
TRANSLATION_FILES=(
  "frontend/src/i18n/"
  "AGENTS.md"
  "README-ES.md"
)

# Crear backup de archivos de traducción
BACKUP_DIR=".sync-backup-$(date +%Y%m%d_%H%M%S)"
echo -e "${BLUE}2. Respaldando archivos de traducción...${NC}"
mkdir -p "$BACKUP_DIR"

for pattern in "${TRANSLATION_FILES[@]}"; do
  if [ -e "$pattern" ]; then
    cp -r "$pattern" "$BACKUP_DIR/" 2>/dev/null || true
  fi
done

# También respaldar archivos Vue modificados (contienen $t())
echo -e "${BLUE}3. Respaldando componentes Vue con traducciones...${NC}"
for file in frontend/src/components/*.vue frontend/src/views/*.vue; do
  if [ -f "$file" ] && grep -q '\$t(' "$file" 2>/dev/null; then
    mkdir -p "$BACKUP_DIR/$(dirname $file)"
    cp "$file" "$BACKUP_DIR/$file"
  fi
done

# Guardar el estado del main.js (configuración de i18n)
if [ -f "frontend/src/main.js" ]; then
  mkdir -p "$BACKUP_DIR/frontend/src"
  cp frontend/src/main.js "$BACKUP_DIR/frontend/src/"
fi

echo -e "${GREEN}✓ Backup creado en: $BACKUP_DIR${NC}"

# Stash de cambios locales no committeados
echo -e "${BLUE}4. Guardando cambios locales no commiteados...${NC}"
STASH_CREATED=false
if [ -n "$(git status --porcelain)" ]; then
  git stash push -m "auto-stash-before-sync-$(date +%Y%m%d_%H%M%S)" --include-untracked
  STASH_CREATED=true
  echo -e "${GREEN}✓ Cambios guardados en stash${NC}"
else
  echo -e "${GREEN}✓ No hay cambios pendientes${NC}"
fi

# Merge del upstream
echo -e "${BLUE}5. Fusionando cambios del upstream...${NC}"
CONFLICTS=false

if ! git merge upstream/main --no-edit; then
  echo -e "${YELLOW}⚠ Conflictos detectados durante el merge${NC}"
  CONFLICTS=true
  
  if [ "$FORCE_LOCAL" = true ]; then
    echo -e "${YELLOW}Resolviendo conflictos favoreciendo cambios locales...${NC}"
    
    # Obtener lista de archivos en conflicto
    CONFLICT_FILES=$(git diff --name-only --diff-filter=U)
    
    for file in $CONFLICT_FILES; do
      # Para archivos de traducción/vistas, usar versión local
      if [[ "$file" == frontend/src/i18n/* ]] || \
         [[ "$file" == frontend/src/components/* ]] || \
         [[ "$file" == frontend/src/views/* ]] || \
         [[ "$file" == "AGENTS.md" ]]; then
        echo -e "  ${YELLOW}Preservando local:${NC} $file"
        git checkout --ours "$file"
        git add "$file"
      else
        # Para el resto, usar versión del upstream
        echo -e "  ${BLUE}Aceptando upstream:${NC} $file"
        git checkout --theirs "$file"
        git add "$file"
      fi
    done
    
    # Completar el merge
    git commit -m "merge: sincronizar con upstream (conflictos resueltos)"
  else
    echo -e "${RED}Resolve los conflictos manualmente y luego ejecuta:${NC}"
    echo "  git add <archivos>"
    echo "  git commit"
    echo ""
    echo -e "${YELLOW}O usa --force para resolver automáticamente favoreciendo tus cambios locales${NC}"
    exit 1
  fi
fi

# Restaurar archivos de traducción respaldados
echo -e "${BLUE}6. Restaurando archivos de traducción...${NC}"

for pattern in "${TRANSLATION_FILES[@]}"; do
  if [ -e "$BACKUP_DIR/$(basename $pattern)" ]; then
    rm -rf "$pattern" 2>/dev/null || true
    cp -r "$BACKUP_DIR/$(basename $pattern)" "$pattern" 2>/dev/null || true
  fi
done

# Restaurar componentes Vue con $t()
for file in frontend/src/components/*.vue frontend/src/views/*.vue; do
  if [ -f "$BACKUP_DIR/$file" ]; then
    cp "$BACKUP_DIR/$file" "$file"
    echo -e "  ${GREEN}Restaurado:${NC} $file"
  fi
done

# Restaurar main.js
if [ -f "$BACKUP_DIR/frontend/src/main.js" ]; then
  cp "$BACKUP_DIR/frontend/src/main.js" frontend/src/main.js
fi

# Recuperar stash si se creó
if [ "$STASH_CREATED" = true ]; then
  echo -e "${BLUE}7. Recuperando stash...${NC}"
  
  # Ver si hay conflictos al aplicar stash
  if ! git stash pop; then
    echo -e "${YELLOW}⚠ Conflictos al recuperar stash. Resolviendo...${NC}"
    
    # Para archivos de traducción, favorecer versión local
    for file in $(git diff --name-only --diff-filter=U 2>/dev/null); do
      if [[ "$file" == frontend/src/i18n/* ]] || \
         [[ "$file" == frontend/src/components/* ]] || \
         [[ "$file" == frontend/src/views/* ]]; then
        git checkout --ours "$file"
        git add "$file"
      fi
    done
    
    # Limpiar stash
    git stash drop
  fi
fi

# Status final
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     ✓ SINCRONIZACIÓN COMPLETADA                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Archivos preservados:${NC}"
echo "  - frontend/src/i18n/ (traducciones)"
echo "  - Componentes Vue con \$t()"
echo "  - AGENTS.md"
echo ""
echo -e "${YELLOW}Backup disponible en:${NC} $BACKUP_DIR"
echo -e "${YELLOW}Para limpiar:${NC} rm -rf $BACKUP_DIR"
echo ""
echo -e "${BLUE}Estado actual:${NC}"
git status --short

echo ""
echo -e "${BLUE}Próximos commits del upstream:${NC}"
git log HEAD..upstream/main --oneline 2>/dev/null || echo "Actualizado"