# 🐟 MiroFish - Scripts y Docker

## Estructura de directorios

```
docker/
└── graphiti/
    └── docker-compose.yml    # Configuración de Neo4j para Graphiti

scripts/
├── iniciar.sh              # Inicia MiroFish según MEMORY_BACKEND
└── detener.sh             # Detiene todos los servicios
```

## 🚀 Iniciar MiroFish

### 1. Configurar `.env`

```bash
# Copiar el ejemplo
cp .env.example .env

# Editar .env y configurar:
# - MEMORY_BACKEND=zep o graphiti
# - LLM_API_KEY
# - Si MEMORY_BACKEND=graphiti: configurar NEO4J_PASSWORD
```

### 2. Ejecutar script de inicio

```bash
./scripts/iniciar.sh
```

El script detecta automáticamente `MEMORY_BACKEND` del `.env` e inicia los servicios necesarios:

#### Si `MEMORY_BACKEND=zep`:
- ✅ Backend Flask (http://localhost:5001)
- ✅ Frontend Vite (http://localhost:3000)
- ℹ️  Zep Cloud (servicio remoto, no requiere Docker)

#### Si `MEMORY_BACKEND=graphiti`:
- ✅ Backend Flask (http://localhost:5001)
- ✅ Frontend Vite (http://localhost:3000)
- 🐳 Neo4j (http://localhost:7474, bolt://localhost:7687)

## ⏹️  Detener MiroFish

```bash
./scripts/detener.sh
```

Detiene:
- Backend Flask
- Frontend Vite
- Neo4j (si MEMORY_BACKEND=graphiti)

## 📋 Logs

```bash
# Backend
tail -f /tmp/mirofish-backend.log

# Frontend
tail -f /tmp/mirofish-frontend.log

# Neo4j Docker
docker logs mirofish-neo4j -f
```

## 🔧 Solución de problemas

### Neo4j no inicia
```bash
# Verificar logs de Neo4j
docker logs mirofish-neo4j

# Recrear contenedor y volumen (elimina datos)
docker compose -f docker/graphiti/docker-compose.yml down -v
docker compose -f docker/graphiti/docker-compose.yml up -d
```

### Puerto en uso
```bash
# Encontrar proceso usando puerto 5001 o 3000
lsof -i :5001
lsof -i :3000

# Matar proceso
kill -9 <PID>
```

## 📦 Volumenes Docker

Los datos de Neo4j se persisten en volúmenes Docker:

```bash
# Listar volúmenes
docker volume ls | grep mirofish

# Eliminar volumen (borra todos los datos)
docker volume rm <volume_name>
```

## 🔄 Cambiar de Zep a Graphiti

1. Editar `.env`:
   ```
   MEMORY_BACKEND=graphiti
   NEO4J_PASSWORD=tu_contraseña_segura
   ```

2. Reiniciar:
   ```bash
   ./scripts/detener.sh
   ./scripts/iniciar.sh
   ```

## 📝 Notas importantes

- **Zep**: Servicio SaaS remoto, no requiere Docker
- **Graphiti**: Requiere Neo4j local vía Docker
- Los reportes existentes siguen funcionando (archivos estáticos)
- Para chat con reportes viejos, necesitas migrar grafos de Zep a Neo4j
