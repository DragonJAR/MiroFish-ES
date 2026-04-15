"""
Wrapper para el backend Graphiti (v0.28.x)
Implementación de MemoryBackend usando Graphiti con Neo4j

API v0.28.x - Graphiti 0.28.2 (última estable)
Documentación: https://help.getzep.com/graphiti/

NOTA: En v0.28.x la API se simplificó drasticamente:
- No hay graphiti.nodes.* ni graphiti.edges.*
- Todo se accede via graphiti.search() y graphiti.add_episode()
- Los índices se construyen via graphiti.build_indices_and_constraints()
"""

import asyncio
import concurrent.futures
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .base import (
    MemoryBackend,
    SearchResult,
    EntityNode,
    GraphInfo,
    EpisodeResult,
    FilteredEntities,
)


def _sanitize_neo4j_value(v):
    """Convertir valores no-JSON-serializables de Neo4j a tipos nativos de Python."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [_sanitize_neo4j_value(i) for i in v]
    if isinstance(v, dict):
        return {k: _sanitize_neo4j_value(val) for k, val in v.items()}

    # Handle Neo4j temporal and spatial types
    try:
        from neo4j.time import DateTime, Date, Time, Duration

        if isinstance(v, (DateTime, Date, Time, Duration)):
            return v.iso_format() if hasattr(v, "iso_format") else str(v)
    except ImportError:
        pass

    return str(v)  # Fallback for other types (bytes, etc.)


def _sanitize_neo4j_attributes(attrs):
    """Sanitizar dict de attributes de Neo4j para que sea JSON-serializable."""
    if not attrs or not isinstance(attrs, dict):
        return attrs or {}
    return {k: _sanitize_neo4j_value(v) for k, v in attrs.items()}


from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.memory.graphiti_backend")

# ── Event Loop Dedicado (Thread-Safe) ──────────────────────────────
# Graphiti + Neo4j async driver mantienen referencias internas al loop
# de asyncio. Si se usa un loop distinto por thread, explotan con:
#   RuntimeError: Task got Future attached to a different loop
#
# Esto pasa porque:
#   - _ensure_indices() corre en el thread principal (Flask request)
#   - add_episode() corre en un thread background (TaskManager)
#
# Solución: UN SOLO event loop corriendo en su propio thread daemon.
# Todas las llamadas a _run_async() programan coroutines ahí via
# run_coroutine_threadsafe(), sin importar desde qué thread se llamen.

_loop = None
_loop_thread = None


def _get_shared_loop():
    """Obtener (o crear) el event loop compartido en thread dedicado."""
    global _loop, _loop_thread
    if _loop is None:
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=_loop.run_forever, daemon=True, name="graphiti-event-loop"
        )
        _loop_thread.start()
    return _loop


def _run_async(coro, timeout: float = 3600.0):
    """
    Ejecutar coroutine async en contexto sync.

    Programa la coroutine en el event loop compartido (que corre en un thread
    dedicado) y bloquea hasta que termine. Esto garantiza que TODAS las
    operaciones de Graphiti/Neo4j usen el mismo event loop, sin importar
    desde qué thread se llamen.

    Args:
        coro: La coroutine a ejecutar
        timeout: Tiempo máximo en segundos para esperar el resultado (default: 3600)

    Raises:
        TimeoutError: Si la coroutine no completa en el tiempo especificado
        Exception: Cualquier excepción raised por la coroutine
    """
    loop = _get_shared_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.error(f"Timeout ({timeout}s) esperando resultado de async operation")
        raise TimeoutError(f"Async operation timed out after {timeout}s")


class GraphitiBackend(MemoryBackend):
    """
    Backend Graphiti v0.28.x

    Graphiti 0.28.x maneja la creación de clientes internamente.
    Solo requiere la conexión a Neo4j y configura automáticamente
    el LLM, embedder y cross-encoder usando variables de entorno.

    API disponible en v0.28.x:
    - graphiti.add_episode() — agregar datos
    - graphiti.search() — búsqueda híbrida (semantic + BM25 + graph)
    - graphiti.build_indices_and_constraints() — crear índices
    - graphiti.close() — cerrar conexión
    """

    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
    ):
        self.neo4j_uri = neo4j_uri or getattr(
            Config, "NEO4J_URI", "bolt://localhost:7687"
        )
        self.neo4j_user = neo4j_user or getattr(Config, "NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or getattr(
            Config, "NEO4J_PASSWORD", "mirofish.dragonjar"
        )

        # Configurar variables de entorno para Graphiti
        self._setup_graphiti_env()

        # Inicializar Graphiti (lazy loading)
        self._graphiti = None
        self._indices_built = False

        logger.info("GraphitiBackend inicializado (v0.28.x)")
        logger.info(f"Neo4j: {self.neo4j_uri}")

    def _setup_graphiti_env(self):
        """
        Configurar variables de entorno para Graphiti antes de importar

        Graphiti 0.28.x detecta automáticamente OPENAI_API_KEY,
        OPENAI_BASE_URL y OPENAI_MODEL_NAME de las variables de entorno.
        """
        import os

        os.environ["OPENAI_API_KEY"] = Config.LLM_API_KEY
        os.environ["OPENAI_BASE_URL"] = Config.LLM_BASE_URL

        if hasattr(Config, "LLM_MODEL_NAME"):
            os.environ["OPENAI_MODEL_NAME"] = Config.LLM_MODEL_NAME

    def _get_graphiti(self):
        """Obtener instancia de Graphiti (lazy initialization)"""
        if self._graphiti is None:
            try:
                from graphiti_core import Graphiti
                from graphiti_core.llm_client.config import LLMConfig
                from .hf_embedder import HuggingFaceEmbedder
                from .zhipu_llm_client import ZhipuAILLMClient

                # ── LLM Client ──
                # ZhipuAILLMClient: wrapper de OpenAIGenericClient que limpia
                # las respuestas JSON de z.ai (viene envuelto en ```json```)
                # y maneja content vacío por reasoning exhaustivo.
                llm_config = LLMConfig(
                    api_key=Config.LLM_API_KEY,
                    model=Config.LLM_MODEL_NAME,
                    base_url=Config.LLM_BASE_URL,
                    max_tokens=16384,
                )
                llm_client = ZhipuAILLMClient(config=llm_config)

                # ── Embedder (HuggingFace local) ──
                # z.ai NO tiene modelos de embeddings disponibles.
                # Usamos sentence-transformers local con GPU/MPS aceleración.
                # Configurable vía HF_EMBEDDING_MODEL en .env
                embedder = HuggingFaceEmbedder()

                self._graphiti = Graphiti(
                    uri=self.neo4j_uri,
                    user=self.neo4j_user,
                    password=self.neo4j_password,
                    llm_client=llm_client,
                    embedder=embedder,
                )

                logger.info("Graphiti v0.28.x inicializado:")
                logger.info(f"  LLM: {Config.LLM_MODEL_NAME} via {Config.LLM_BASE_URL}")
                logger.info(
                    f"  Embedder: HuggingFace local ({embedder.model_name}, dim={embedder.embedding_dim})"
                )

            except ImportError as e:
                logger.error(f"No se pudo importar graphiti_core: {e}")
                raise ValueError(
                    "graphiti-core no está instalado. Instale con: uv add graphiti-core"
                )
            except Exception as e:
                logger.error(f"Error al inicializar Graphiti: {e}")
                raise

        return self._graphiti

    @property
    def graph(self):
        """Alias for _get_graphiti() for compatibility with code expecting .graph attribute"""
        return self._get_graphiti()

    def _ensure_indices(self):
        """Construir índices en primer uso"""
        if not self._indices_built:
            graphiti = self._get_graphiti()

            # =========================================================
            # 1. build_indices_and_constraints() — puede lanzar
            #    EquivalentSchemaRuleAlreadyExistsException si el índice
            #    ya existe (bug conocido en Graphiti 0.28.x)
            # =========================================================
            try:
                _run_async(graphiti.build_indices_and_constraints())
            except Exception as build_err:
                # Si el error es "index already exists", es OK — continuamos
                error_str = str(build_err)
                if (
                    "EquivalentSchemaRuleAlreadyExistsException" in error_str
                    or "already exists" in error_str.lower()
                ):
                    logger.info(
                        "Índices de Graphiti ya existen (EquivalentSchemaRuleAlreadyExistsException), continuando..."
                    )
                else:
                    logger.warning(
                        f"No se pudieron construir índices de Graphiti: {build_err}"
                    )

            # =========================================================
            # 2. ÍNDICES VECTORIALES NECESARIOS PARA Neo4j 5 Community
            # =========================================================
            # Graphiti internamente usa vector.similarity.cosine() para
            # deduplicar entidades en add_episode(). Neo4j 5 Community
            # requiere un VECTOR INDEX en la propiedad para que
            # vector.similarity.cosine() funcione con LIST<FLOAT>.
            # Sin este índice, add_episode() falla con:
            # "Invalid input for 'vector.similarity.cosine()':
            #  Argument b is not a valid vector"
            #
            # Graphiti.build_indices_and_constraints() NO crea este índice.
            try:
                self._create_vector_index(graphiti)
                self._indices_built = True
                logger.info("Índices de Graphiti construidos")
            except Exception as idx_err:
                logger.warning(f"No se pudieron crear índices vectoriales: {idx_err}")

    def _create_vector_index(self, graphiti):
        """
        Crear índice vectorial en Entity.name_embedding.

        Neo4j 5 Community necesita un vector index para que
        vector.similarity.cosine() funcione correctamente.

        El índice se crea con:
        - Nombre: entity_name_embedding_idx
        - Solo en nodos Entity
        - Propiedad: name_embedding
        - Dimensión: según HF_EMBEDDING_MODEL (default all-MiniLM-L6-v2=384)
        - Función: cosine
        """
        import os

        # Pre-check: Verificar si el índice ya existe antes de intentar crearlo
        # Esto evita el error EquivalentSchemaRuleAlreadyExistsException
        try:
            check_cypher = "SHOW INDEXES YIELD name, type, labelsOrTypes, properties WHERE name = 'entity_name_embedding_idx' RETURN name"
            result = _run_async(graphiti.driver.execute_query(check_cypher))
            result_data = _run_async(result.data())
            if result_data and len(result_data) > 0:
                logger.info(
                    "Vector index 'entity_name_embedding_idx' ya existe, omitiendo creación"
                )
                return True
        except Exception as check_err:
            # Si el check falla, logueamos pero continuamos con la creación
            # (el except de abajo capturará "already exists" si es el caso)
            logger.debug(
                f"Index check query no retornó resultados o falló: {check_err}"
            )

        try:
            # Obtener dimensión del embedding desde el modelo configurado
            from .hf_embedder import HuggingFaceEmbedder

            model_name = os.environ.get("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            dim = HuggingFaceEmbedder.MODELS.get(model_name, 384)

            # Cypher para crear el índice vectorial con sintaxis de procedimiento
            # Neo4j 5.26+ requiere CALL db.index.vector.createNodeIndex(...) en lugar
            # de la sintaxis declarativa CREATE VECTOR INDEX ... OPTIONS
            cypher = f"""CALL db.index.vector.createNodeIndex(
              'entity_name_embedding_idx',
              'Entity',
              'name_embedding',
              {dim},
              'cosine'
            )"""

            _run_async(graphiti.driver.execute_query(cypher))
            logger.info(
                f"Índice vectorial entity_name_embedding_idx creado (dim={dim})"
            )

        except Exception as e:
            # Bug B fix: Handle "index already exists" error specifically
            error_str = str(e)
            if (
                "EquivalentSchemaRuleAlreadyExistsException" in error_str
                or "already exists" in error_str.lower()
            ):
                index_name = "entity_name_embedding_idx"
                logger.info(
                    f"Vector index '{index_name}' ya existe, omitiendo creación"
                )
                return True
            # No crashar si el índice no se puede crear
            # (puede fallar si Neo4j no soporta la sintaxis o hay otro error)
            logger.warning(f"No se pudo crear índice vectorial: {e}")

    def search(
        self,
        query: str,
        graph_id: str,
        mode: str = "quick",
        limit: int = 10,
    ) -> SearchResult:
        """
        Búsqueda en el grafo Graphiti usando search().

        EntityEdge solo tiene source_node_uuid/target_node_uuid (strings),
        NO objetos nodo. Obtenemos nombres via Cypher JOIN.
        """
        logger.info(
            f"Búsqueda Graphiti: graph_id={graph_id}, query={query[:50]}..., mode={mode}"
        )

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            search_results = _run_async(
                graphiti.search(
                    query=query,
                    group_ids=[graph_id],
                    num_results=limit,
                )
            )

            facts = []
            edges = []
            seen_node_uuids = set()
            nodes = []

            # Recopilar todos los UUIDs de nodos que necesitamos resolver
            for edge in search_results:
                fact = getattr(edge, "fact", "")
                if fact:
                    facts.append(fact)

                src_uuid = getattr(edge, "source_node_uuid", "")
                tgt_uuid = getattr(edge, "target_node_uuid", "")
                if src_uuid:
                    seen_node_uuids.add(src_uuid)
                if tgt_uuid:
                    seen_node_uuids.add(tgt_uuid)

                edges.append(
                    {
                        "uuid": getattr(edge, "uuid", ""),
                        "name": getattr(edge, "name", ""),
                        "fact": fact,
                        "source_node_uuid": src_uuid,
                        "target_node_uuid": tgt_uuid,
                    }
                )

            # Resolver nombres de nodos en batch via Cypher (una sola query)
            if seen_node_uuids:
                uuid_list = list(seen_node_uuids)
                node_query = """
                MATCH (n:Entity)
                WHERE n.uuid IN $uuids
                RETURN n.uuid AS uuid, n.name AS name,
                       labels(n) AS labels,
                       n.summary AS summary
                """
                node_result = _run_async(
                    graphiti.driver.execute_query(node_query, uuids=uuid_list)
                )
                node_map = {}
                for record in node_result.records:
                    node_map[record["uuid"]] = record

                # Bug 3 fix: deduplicate nodes — same UUID can appear as source and target
                nodes_seen = set()
                for edge_dict in edges:
                    for key in ["source_node_uuid", "target_node_uuid"]:
                        uuid = edge_dict[key]
                        if uuid in node_map and uuid not in nodes_seen:
                            r = node_map[uuid]
                            nodes.append(
                                {
                                    "uuid": r["uuid"],
                                    "name": r["name"],
                                    "labels": r.get("labels", []),
                                    "summary": r.get("summary", ""),
                                }
                            )
                            nodes_seen.add(uuid)

            logger.info(f"Búsqueda completada: {len(facts)} hechos, {len(nodes)} nodos")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )

        except Exception as e:
            logger.error(f"Búsqueda Graphiti falló: {str(e)}")
            return SearchResult(
                facts=[], edges=[], nodes=[], query=query, total_count=0
            )

    def get_entities(
        self,
        graph_id: str,
        entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """
        Obtener entidades del grafo Graphiti via Cypher directo.

        Graphiti v0.28.x NO usa labels de entity_type en Neo4j (solo pone "Entity").
        Por eso no filtramos por custom_labels — retornamos TODOS los nodos Entity
        del grupo y usamos "Entity" como entity_type.
        """
        logger.info(f"Obteniendo entidades de grafo {graph_id}...")

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Query 1: Get all entities
            query_entities = """
            MATCH (n:Entity {group_id: $group_id})
            RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                   n.summary AS summary, n.created_at AS created_at,
                   properties(n) AS attributes
            """

            result = _run_async(
                graphiti.driver.execute_query(query_entities, group_id=graph_id)
            )

            # Build UUID → entity map
            entity_map = {}
            for record in result.records:
                entity_map[record["uuid"]] = {
                    "uuid": record.get("uuid", ""),
                    "name": record.get("name", ""),
                    "labels": record.get("labels", []) or [],
                    "summary": record.get("summary", ""),
                    "attributes": record.get("attributes", {}),
                }

            # Query 2: Get all edges between entities in this group
            query_edges = """
            MATCH (src:Entity {group_id: $group_id})-[r]-(tgt:Entity {group_id: $group_id})
            RETURN src.uuid AS source_uuid, tgt.uuid AS target_uuid,
                   type(r) AS edge_name,
                   properties(r) AS rel_props
            """

            try:
                edge_result = _run_async(
                    graphiti.driver.execute_query(query_edges, group_id=graph_id)
                )

                # Build adjacency: uuid → list of {source_uuid, target_uuid, edge_name, direction}
                outgoing = {}  # source_uuid → [edges]
                incoming = {}  # target_uuid → [edges]

                for rec in edge_result.records:
                    src = rec.get("source_uuid", "")
                    tgt = rec.get("target_uuid", "")
                    ename = rec.get("edge_name", "") or ""
                    rprops = rec.get("rel_props") or {}
                    fact = rprops.get("fact", "")

                    edge_out = {
                        "fact": fact,
                        "edge_name": ename,
                        "direction": "outgoing",
                        "target_uuid": tgt,
                    }
                    edge_in = {
                        "fact": fact,
                        "edge_name": ename,
                        "direction": "incoming",
                        "source_uuid": src,
                    }

                    outgoing.setdefault(src, []).append(edge_out)
                    incoming.setdefault(tgt, []).append(edge_in)
            except Exception as e:
                logger.warning(f"No se pudieron obtener edges: {e}")
                outgoing = {}
                incoming = {}

            # Build EntityNode objects with populated related_edges and related_nodes
            entities = []
            for uuid, data in entity_map.items():
                # Collect related edges (outgoing + incoming)
                rel_edges = outgoing.get(uuid, []) + incoming.get(uuid, [])

                # Collect related nodes (unique, excluding self)
                seen_nodes = set()
                rel_nodes = []
                for edge in rel_edges:
                    other_uuid = edge.get("target_uuid") or edge.get("source_uuid", "")
                    if (
                        other_uuid
                        and other_uuid != uuid
                        and other_uuid not in seen_nodes
                    ):
                        seen_nodes.add(other_uuid)
                        other = entity_map.get(other_uuid)
                        if other:
                            rel_nodes.append(
                                {
                                    "name": other["name"],
                                    "labels": other["labels"],
                                    "summary": other["summary"],
                                }
                            )

                entity = EntityNode(
                    uuid=data["uuid"],
                    name=data["name"],
                    labels=data["labels"],
                    summary=data["summary"],
                    attributes=_sanitize_neo4j_attributes(data["attributes"]),
                    related_edges=rel_edges,
                    related_nodes=rel_nodes,
                )
                entities.append(entity)

            logger.info(f"Obtenidas {len(entities)} entidades")
            return entities

        except Exception as e:
            logger.error(f"Error al obtener entidades: {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """
        Filtrar nodos que coincidan con tipos de entidad predefinidos.

        Logica de filtrado:
        - Si defined_entity_types esta proporcionado: mantener solo entidades
          cuyas labels intersecten con la lista
        - Si defined_entity_types es None/vacio: mantener entidades cuyas labels
          contengan algo DISTINTO de solo "Entity" y "Node"

        Args:
            graph_id: ID del grafo
            defined_entity_types: Lista de tipos de entidad predefinidos (opcional)
            enrich_with_edges: Si incluir informacion de bordes relacionados

        Returns:
            FilteredEntities: Conjunto de entidades filtradas
        """
        logger.info(f"Iniciar filtrado de entidades del grafo {graph_id}...")

        # Obtener todas las entidades (ya viene con related_edges y related_nodes)
        all_entities = self.get_entities(graph_id=graph_id)
        total_count = len(all_entities)

        # Filtrar entidades segun logica de defined_entity_types
        filtered_entities = []
        entity_types_found = set()

        for entity in all_entities:
            labels = entity.labels or []

            # Obtener custom labels (excluir "Entity" y "Node")
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

            if not custom_labels:
                # No tiene etiquetas custom — derivar tipo del nombre
                # Patrones conocidos: "Expert", "Leader", "Analyst", "Manager", "Specialist"
                name_lower = entity.name.lower()
                if "leader" in name_lower:
                    entity_type = "OpinionLeader"
                elif "expert" in name_lower:
                    entity_type = "Expert"
                elif "analyst" in name_lower:
                    entity_type = "Analyst"
                elif "manager" in name_lower:
                    entity_type = "Manager"
                elif "specialist" in name_lower:
                    entity_type = "Specialist"
                elif "researcher" in name_lower:
                    entity_type = "Researcher"
                elif "coordinator" in name_lower:
                    entity_type = "Coordinator"
                else:
                    entity_type = "Agent"
            elif defined_entity_types:
                # Filtrar por tipos predefinidos
                matching_labels = [
                    l for l in custom_labels if l in defined_entity_types
                ]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)
            filtered_entities.append(entity)

        logger.info(
            f"Filtrado completado: Total {total_count}, filtradas {len(filtered_entities)}, "
            f"tipos encontrados: {entity_types_found}"
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_by_uuid(
        self,
        graph_id: str,
        uuid: str,
    ) -> Optional[EntityNode]:
        """
        Obtener una entidad por UUID via Cypher directo.

        NOTA: graphiti._search() usa vector.similarity.cosine() que NO funciona
        en Neo4j 5 Community. Usamos Cypher directo para evitar el problema.
        """
        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Cypher directo para obtener entidad por uuid
            query = """
            MATCH (n:Entity {group_id: $group_id, uuid: $uuid})
            RETURN n.uuid AS uuid, n.name AS name, labels(n) AS labels,
                   n.summary AS summary, n.created_at AS created_at,
                   properties(n) AS attributes
            """

            result = _run_async(
                graphiti.driver.execute_query(query, group_id=graph_id, uuid=uuid)
            )

            if not result or not result.records:
                return None
            record = result.records[0]

            return EntityNode(
                uuid=record.get("uuid", ""),
                name=record.get("name", ""),
                labels=record.get("labels", []),
                summary=record.get("summary", ""),
                attributes=_sanitize_neo4j_attributes(record.get("attributes", {})),
            )

        except Exception as e:
            logger.error(f"Error al obtener entidad {uuid}: {str(e)}")
            return None

    def get_edges(
        self,
        graph_id: str,
        entity_uuid: Optional[str] = None,
        include_temporal: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Obtener bordes del grafo Graphiti via Cypher directo.

        NOTA: graphiti.search() y graphiti._search() usan vector.similarity.cosine()
        que NO funciona en Neo4j 5 Community. Usamos Cypher directo para evitar
        el problema y obtener todos los bordes RELATES_TO del grupo.
        """
        logger.info(f"Obteniendo bordes de grafo {graph_id}...")

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Cypher directo para obtener todas las relaciones RELATES_TO del grupo
            # Incluye nombres de nodos source/target via MATCH
            query = """
            MATCH (source:Entity {group_id: $group_id})-[r:RELATES_TO]->(target:Entity {group_id: $group_id})
            RETURN r.uuid AS uuid, r.name AS name, r.fact AS fact,
                   source.uuid AS source_node_uuid, source.name AS source_node_name,
                   target.uuid AS target_node_uuid, target.name AS target_node_name,
                   r.created_at AS created_at, r.valid_at AS valid_at, r.invalid_at AS invalid_at,
                   properties(r) AS attributes
            """

            result = _run_async(graphiti.driver.execute_query(query, group_id=graph_id))

            edges_data = []
            for record in result.records:
                source_node_uuid = record.get("source_node_uuid", "")
                target_node_uuid = record.get("target_node_uuid", "")

                # Filtrar por entity_uuid si se proporciona
                if entity_uuid:
                    if (
                        source_node_uuid != entity_uuid
                        and target_node_uuid != entity_uuid
                    ):
                        continue

                edge_dict = {
                    "uuid": record.get("uuid", ""),
                    "name": record.get("name", ""),
                    "fact": record.get("fact", ""),
                    "source_node_uuid": source_node_uuid,
                    "source_node_name": record.get("source_node_name", ""),
                    "target_node_uuid": target_node_uuid,
                    "target_node_name": record.get("target_node_name", ""),
                    "attributes": _sanitize_neo4j_attributes(
                        record.get("attributes", {})
                    ),
                }

                if include_temporal:
                    edge_dict["created_at"] = _sanitize_neo4j_value(
                        record.get("created_at", None)
                    )
                    edge_dict["valid_at"] = _sanitize_neo4j_value(
                        record.get("valid_at", None)
                    )
                    edge_dict["invalid_at"] = _sanitize_neo4j_value(
                        record.get("invalid_at", None)
                    )

                edges_data.append(edge_dict)

            logger.info(f"Obtenidos {len(edges_data)} bordes")
            return edges_data

        except Exception as e:
            logger.error(f"Error al obtener bordes: {str(e)}")
            return []

    def add_episode(
        self,
        graph_id: str,
        content: str,
        reference_time: Optional[str] = None,
        name: Optional[str] = None,
        source_type: str = "text",
    ) -> EpisodeResult:
        """
        Agregar episodio al grafo Graphiti

        API v0.28.x: add_episode(name, episode_body, source_description,
        reference_time, source, group_id)
        """
        logger.info(f"Agregando episodio a grafo {graph_id}...")

        max_timeout_retries = 2
        last_error = None
        for timeout_attempt in range(max_timeout_retries + 1):
            try:
                self._ensure_indices()
                graphiti = self._get_graphiti()

                from graphiti_core.nodes import EpisodeType

                # Convertir reference_time a datetime si es string
                ref_time = None
                if reference_time:
                    try:
                        ref_time = datetime.fromisoformat(reference_time)
                        if ref_time.tzinfo is None:
                            ref_time = ref_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        ref_time = datetime.now(timezone.utc)

                episode_name = name or f"Episode_{ref_time or 'now'}"

                result = _run_async(
                    graphiti.add_episode(
                        name=episode_name,
                        episode_body=content,
                        source_description=source_type,
                        reference_time=ref_time or datetime.now(timezone.utc),
                        source=EpisodeType.message,
                        group_id=graph_id,
                    )
                )

                episode_uuid = getattr(result, "uuid", "")

                logger.info(f"Episodio agregado: {episode_uuid}")

                return EpisodeResult(
                    episode_uuid=episode_uuid,
                    status="completed",
                )

            except TimeoutError as e:
                last_error = e
                if timeout_attempt < max_timeout_retries:
                    logger.warning(
                        f"Timeout al agregar episodio (intento {timeout_attempt + 1}/{max_timeout_retries + 1}), reintentando..."
                    )
                    continue
                logger.error(
                    f"Timeout después de {max_timeout_retries + 1} intentos: {e}"
                )
                raise

            except Exception as e:
                logger.error(f"Error al agregar episodio: {str(e)}")
                raise

    def create_graph(
        self,
        name: str,
        ontology: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Crear nuevo grafo en Graphiti

        Graphiti 0.28.x crea grafos implícitamente con el primer episodio
        """
        import uuid as uuid_lib

        graph_id = f"mirofish_{uuid_lib.uuid4().hex[:16]}"

        logger.info(f"Grafo Graphiti preparado: {graph_id}")
        return graph_id

    def delete_graph(self, graph_id: str) -> bool:
        """
        Eliminar grafo Graphiti

        v0.28.x: eliminar todos los datos del grupo via driver Cypher
        """
        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Usar Cypher directo para eliminar nodos del grupo
            query = """
            MATCH (n {group_id: $group_id})
            DETACH DELETE n
            """
            _run_async(graphiti.driver.execute_query(query, group_id=graph_id))

            logger.info(f"Grafo eliminado: {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error al eliminar grafo {graph_id}: {str(e)}")
            return False

    def build_indices(self) -> bool:
        """Construir índices de Neo4j para Graphiti"""
        try:
            self._ensure_indices()
            return True
        except Exception as e:
            logger.error(f"Error al construir índices: {str(e)}")
            return False

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Get graph statistics using Cypher queries to Neo4j"""

        async def _get_stats():
            async with self.graph.driver.session() as session:
                # Count nodes
                node_result = await session.run(
                    "MATCH (n:Entity) WHERE n.group_id = $graph_id RETURN count(n) as total_nodes",
                    graph_id=graph_id,
                )
                node_count = (await node_result.single())["total_nodes"]

                # Count edges
                edge_result = await session.run(
                    "MATCH ()-[r]->() WHERE r.group_id = $graph_id RETURN count(r) as total_edges",
                    graph_id=graph_id,
                )
                edge_count = (await edge_result.single())["total_edges"]

                # Get entity types
                type_result = await session.run(
                    "MATCH (n:Entity) WHERE n.group_id = $graph_id UNWIND labels(n) as label RETURN collect(distinct label) as entity_types",
                    graph_id=graph_id,
                )
                entity_types = (await type_result.single())["entity_types"]

                # Get relation types
                rel_result = await session.run(
                    "MATCH ()-[r]->() WHERE r.group_id = $graph_id UNWIND type(r) as rel_type RETURN collect(distinct rel_type) as relation_types",
                    graph_id=graph_id,
                )
                relation_types = (await rel_result.single())["relation_types"]

                return {
                    "graph_id": graph_id,
                    "total_nodes": node_count,
                    "total_edges": edge_count,
                    "entity_types": entity_types,
                    "relation_types": relation_types,
                }

        return asyncio.run(_get_stats())

    def get_entities_by_type(
        self, graph_id: str, entity_type: str
    ) -> List[Dict[str, Any]]:
        """Get entities by type using Cypher"""

        async def _get_entities():
            async with self.graph.driver.session() as session:
                result = await session.run(
                    "MATCH (n:Entity) WHERE n.group_id = $graph_id AND $entity_type IN labels(n) "
                    "RETURN n.uuid as uuid, n.name as name, n.entity_type as entity_type, n.created_at as created_at",
                    graph_id=graph_id,
                    entity_type=entity_type,
                )
                records = await result.data()
                return [
                    {
                        "uuid": r["uuid"],
                        "name": r["name"],
                        "entity_type": r["entity_type"],
                        "created_at": r["created_at"],
                    }
                    for r in records
                ]

        return asyncio.run(_get_entities())

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """Get entity + its relationships"""

        async def _get_summary():
            async with self.graph.driver.session() as session:
                # Get entity
                node_result = await session.run(
                    "MATCH (n:Entity {group_id: $graph_id, name: $entity_name}) "
                    "OPTIONAL MATCH (n)-[r]-() RETURN n as entity, collect(r) as relationships",
                    graph_id=graph_id,
                    entity_name=entity_name,
                )
                record = await node_result.single()
                if not record or not record["entity"]:
                    return {"entity": None, "relationships": []}

                entity = record["entity"]
                relationships = record["relationships"]

                return {
                    "entity": {
                        "name": entity.get("name"),
                        "entity_type": entity.get("entity_type"),
                        "uuid": entity.get("uuid"),
                    },
                    "relationships": [
                        {
                            "type": type(r),
                            "target": r.end_node.get("name") if r.end_node else None,
                        }
                        for r in relationships
                        if r
                    ],
                }

        return asyncio.run(_get_summary())

    def get_simulation_context(
        self, graph_id: str, simulation_requirement: str, limit: int = 10
    ) -> Dict[str, Any]:
        """Get statistics + LLM summary for simulation context"""
        from ..utils.llm_client import LLMClient

        # Get basic stats
        stats = self.get_graph_statistics(graph_id)

        # Get top entities
        entities_result = self.get_entities_by_type(graph_id, "Agent")
        top_agents = entities_result[:limit]

        # Build context for LLM
        context = {
            "graph_id": graph_id,
            "statistics": stats,
            "top_agents": [
                {"name": a.get("name"), "type": a.get("entity_type")}
                for a in top_agents
            ],
            "simulation_requirement": simulation_requirement,
        }

        # Generate LLM summary
        llm_client = LLMClient()
        prompt = f"""Analyze this simulation graph and provide a summary:

Graph Statistics:
- Total Nodes: {stats.get("total_nodes", 0)}
- Total Edges: {stats.get("total_edges", 0)}
- Entity Types: {", ".join(stats.get("entity_types", []))}
- Relation Types: {", ".join(stats.get("relation_types", []))}

Top Agents:
{chr(10).join([f"- {a.get('name')}: {a.get('entity_type')}" for a in top_agents])}

Simulation Requirement: {simulation_requirement}

Provide a concise summary of the simulation context."""

        try:
            summary = llm_client.chat(prompt)
            context["llm_summary"] = summary
        except Exception as e:
            context["llm_summary"] = f"LLM summary unavailable: {str(e)}"

        return context
