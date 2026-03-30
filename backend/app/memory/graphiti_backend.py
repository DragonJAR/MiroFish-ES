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

from .base import MemoryBackend, SearchResult, EntityNode, GraphInfo, EpisodeResult
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


def _run_async(coro, timeout: float = 60.0):
    """
    Ejecutar coroutine async en contexto sync.

    Programa la coroutine en el event loop compartido (que corre en un thread
    dedicado) y bloquea hasta que termine. Esto garantiza que TODAS las
    operaciones de Graphiti/Neo4j usen el mismo event loop, sin importar
    desde qué thread se llamen.

    Args:
        coro: La coroutine a ejecutar
        timeout: Tiempo máximo en segundos para esperar el resultado (default: 60)

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

    def _ensure_indices(self):
        """Construir índices en primer uso"""
        if not self._indices_built:
            try:
                graphiti = self._get_graphiti()
                _run_async(graphiti.build_indices_and_constraints())
                self._indices_built = True
                logger.info("Índices de Graphiti construidos")
            except Exception as e:
                logger.warning(f"No se pudieron construir índices: {e}")

    def search(
        self,
        query: str,
        graph_id: str,
        mode: str = "quick",
        limit: int = 10,
    ) -> SearchResult:
        """
        Búsqueda en el grafo Graphiti usando search()

        v0.28.x usa group_ids (plural) y retorna EntityEdge objects
        """
        logger.info(
            f"Búsqueda Graphiti: graph_id={graph_id}, query={query[:50]}..., mode={mode}"
        )

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # v0.28.x: search() usa group_ids (plural, lista)
            search_results = _run_async(
                graphiti.search(
                    query=query,
                    group_ids=[graph_id],
                    num_results=limit,
                )
            )

            facts = []
            edges = []
            nodes = []

            # Parsear EntityEdge results de v0.28.x
            for edge in search_results:
                fact = getattr(edge, "fact", "")
                if fact:
                    facts.append(fact)

                # Extraer nodos fuente y destino
                source = getattr(edge, "source_node", None)
                target = getattr(edge, "target_node", None)

                if source:
                    nodes.append(
                        {
                            "uuid": getattr(source, "uuid", ""),
                            "name": getattr(source, "name", ""),
                            "labels": getattr(source, "labels", []),
                            "summary": getattr(source, "summary", ""),
                        }
                    )

                if target:
                    nodes.append(
                        {
                            "uuid": getattr(target, "uuid", ""),
                            "name": getattr(target, "name", ""),
                            "labels": getattr(target, "labels", []),
                            "summary": getattr(target, "summary", ""),
                        }
                    )

                edges.append(
                    {
                        "uuid": getattr(edge, "uuid", ""),
                        "name": getattr(edge, "name", ""),
                        "fact": fact,
                        "source_node_uuid": getattr(source, "uuid", "")
                        if source
                        else "",
                        "target_node_uuid": getattr(target, "uuid", "")
                        if target
                        else "",
                    }
                )

            logger.info(f"Búsqueda completada: {len(facts)} hechos encontrados")

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
        Obtener entidades del grafo Graphiti via search()

        v0.28.x no tiene graphiti.nodes.* — usamos search_() con config de nodos
        """
        logger.info(f"Obteniendo entidades de grafo {graph_id}...")

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            from graphiti_core.search.search_config_recipes import (
                NODE_HYBRID_SEARCH_RRF,
            )

            # Usar recipe de búsqueda de nodos con group_ids
            node_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            node_config.limit = 100 if not entity_types else 50

            results = _run_async(
                graphiti._search(
                    query=graph_id,
                    config=node_config,
                    group_ids=[graph_id],
                )
            )

            entities = []
            for node in results.nodes:
                labels = getattr(node, "labels", [])
                custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

                if not custom_labels:
                    continue

                if entity_types:
                    matching = [l for l in custom_labels if l in entity_types]
                    if not matching:
                        continue

                entity = EntityNode(
                    uuid=getattr(node, "uuid", ""),
                    name=getattr(node, "name", ""),
                    labels=labels,
                    summary=getattr(node, "summary", ""),
                    attributes=getattr(node, "attributes", {}),
                )
                entities.append(entity)

            logger.info(f"Obtenidas {len(entities)} entidades")
            return entities

        except Exception as e:
            logger.error(f"Error al obtener entidades: {str(e)}")
            return []

    def get_entity_by_uuid(
        self,
        graph_id: str,
        uuid: str,
    ) -> Optional[EntityNode]:
        """
        Obtener una entidad por UUID via search()

        v0.28.x no tiene graphiti.nodes.get() — usamos búsqueda directa
        """
        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            from graphiti_core.search.search_config_recipes import (
                NODE_HYBRID_SEARCH_RRF,
            )

            node_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            node_config.limit = 1

            # Buscar nodos que coincidan con el UUID
            results = _run_async(
                graphiti._search(
                    query=uuid,
                    config=node_config,
                    group_ids=[graph_id],
                )
            )

            for node in results.nodes:
                if getattr(node, "uuid", "") == uuid:
                    return EntityNode(
                        uuid=getattr(node, "uuid", ""),
                        name=getattr(node, "name", ""),
                        labels=getattr(node, "labels", []),
                        summary=getattr(node, "summary", ""),
                        attributes=getattr(node, "attributes", {}),
                    )

            return None

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
        Obtener bordes del grafo Graphiti via search()

        v0.28.x no tiene graphiti.edges.* — usamos search()
        """
        logger.info(f"Obteniendo bordes de grafo {graph_id}...")

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Buscar todo en el grupo (sin query específico usa BFS)
            search_results = _run_async(
                graphiti.search(
                    query="*",
                    group_ids=[graph_id],
                    num_results=100,
                )
            )

            edges_data = []
            for edge in search_results:
                source = getattr(edge, "source_node", None)
                target = getattr(edge, "target_node", None)

                edge_dict = {
                    "uuid": getattr(edge, "uuid", ""),
                    "name": getattr(edge, "name", ""),
                    "fact": getattr(edge, "fact", ""),
                    "source_node_uuid": getattr(source, "uuid", "") if source else "",
                    "target_node_uuid": getattr(target, "uuid", "") if target else "",
                    "attributes": getattr(edge, "attributes", {}),
                }

                if include_temporal:
                    edge_dict["created_at"] = getattr(edge, "created_at", None)
                    edge_dict["valid_at"] = getattr(edge, "valid_at", None)
                    edge_dict["invalid_at"] = getattr(edge, "invalid_at", None)

                # Filtrar por entity_uuid si se proporciona
                if entity_uuid:
                    if (
                        edge_dict["source_node_uuid"] != entity_uuid
                        and edge_dict["target_node_uuid"] != entity_uuid
                    ):
                        continue

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
            _run_async(graphiti._driver.execute_query(query, {"group_id": graph_id}))

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
