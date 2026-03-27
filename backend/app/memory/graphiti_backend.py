"""
Wrapper para el backend Graphiti
Implementación de MemoryBackend usando Graphiti con Neo4j
"""

import asyncio
from typing import Dict, Any, List, Optional

from .base import MemoryBackend, SearchResult, EntityNode, GraphInfo, EpisodeResult
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.memory.graphiti_backend")


def _run_async(coro):
    """
    Ejecutar coroutine async en contexto sync

    Necesario para puentear la API async de Graphiti con Flask sync
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class GraphitiBackend(MemoryBackend):
    """
    Backend Graphiti

    Utiliza Graphiti con Neo4j como backend de almacenamiento
    Requiere configuración LLM para extracción de entidades
    """

    def __init__(
        self,
        llm_provider: str = None,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
    ):
        self.llm_provider = (
            llm_provider or Config.GRAPHITI_LLM_PROVIDER
            if hasattr(Config, "GRAPHITI_LLM_PROVIDER")
            else "zai"
        )
        self.neo4j_uri = neo4j_uri or getattr(
            Config, "NEO4J_URI", "bolt://localhost:7687"
        )
        self.neo4j_user = neo4j_user or getattr(Config, "NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or getattr(
            Config, "NEO4J_PASSWORD", "password"
        )

        # Inicializar cliente LLM
        self._llm_client = self._create_llm_client()
        self._embedder = self._create_embedder()

        # Inicializar Graphiti (lazy loading para evitar import temprano)
        self._graphiti = None
        self._indices_built = False

        logger.info("GraphitiBackend inicializado")

    def _create_llm_client(self):
        """Crear cliente LLM usando AsyncOpenAI"""
        from openai import AsyncOpenAI

        if self.llm_provider == "zai":
            return AsyncOpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
            )
        elif self.llm_provider == "minimax":
            return AsyncOpenAI(
                api_key=Config.LLM_FALLBACK_API_KEY,
                base_url=Config.LLM_FALLBACK_BASE_URL,
            )
        else:  # openai
            return AsyncOpenAI(api_key=Config.LLM_API_KEY)

    def _create_embedder(self):
        """Crear embedder para embeddings de texto"""
        from openai import AsyncOpenAI

        # Usar el mismo cliente LLM para embeddings
        return AsyncOpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
        )

    def _get_graphiti(self):
        """Obtener instancia de Graphiti (lazy initialization)"""
        if self._graphiti is None:
            try:
                from graphiti_core import Graphiti
                from graphiti_core.llm_client import OpenAIClient
                # from graphiti_core.utils.migrate_neo4j import migrate_neo4j  # No existe en v0.11.6

                # Crear cliente LLM compatible con Graphiti
                # Graphiti espera un cliente sync, así que necesitamos adaptar
                llm_client = OpenAIClient(
                    api_key=Config.LLM_API_KEY,
                    model=Config.LLM_MODEL_NAME,
                    base_url=Config.LLM_BASE_URL
                    if self.llm_provider != "openai"
                    else None,
                )

                # Inicializar Graphiti
                self._graphiti = Graphiti(
                    uri=self.neo4j_uri,
                    user=self.neo4j_user,
                    password=self.neo4j_password,
                    llm_client=llm_client,
                )

                # Ejecutar migración si es necesario
                # _run_async(
                #     migrate_neo4j(self.neo4j_uri, self.neo4j_user, self.neo4j_password)
                # )  # No existe en v0.11.6

                logger.info("Graphiti inicializado exitosamente")

            except ImportError as e:
                logger.error(f"No se pudo importar graphiti_core: {e}")
                raise ValueError(
                    "graphiti_core no está instalado. Instale con: pip install graphiti_core>=0.3.0"
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
        Búsqueda en el grafo Graphiti

        Utiliza graphiti.search() con parámetros según el modo
        """
        logger.info(
            f"Búsqueda Graphiti: graph_id={graph_id}, query={query[:50]}..., mode={mode}"
        )

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Graphiti usa group_id en lugar de graph_id
            # Para compatibilidad, usamos graph_id como group_id
            group_id = graph_id

            # Ejecutar búsqueda (Graphiti es async)
            search_results = _run_async(
                graphiti.search(
                    query=query,
                    num_results=limit,
                    group_id=group_id,
                )
            )

            facts = []
            edges = []
            nodes = []

            # Parsear resultados de Graphiti
            for result in search_results:
                if hasattr(result, "fact"):
                    facts.append(result.fact)

                # Crear nodos y bordes basados en resultados
                if hasattr(result, "source_node") and result.source_node:
                    nodes.append(
                        {
                            "uuid": getattr(result.source_node, "uuid", ""),
                            "name": getattr(result.source_node, "name", ""),
                            "labels": getattr(result.source_node, "labels", []),
                            "summary": getattr(result.source_node, "summary", ""),
                        }
                    )

                if hasattr(result, "target_node") and result.target_node:
                    nodes.append(
                        {
                            "uuid": getattr(result.target_node, "uuid", ""),
                            "name": getattr(result.target_node, "name", ""),
                            "labels": getattr(result.target_node, "labels", []),
                            "summary": getattr(result.target_node, "summary", ""),
                        }
                    )

                edges.append(
                    {
                        "uuid": getattr(result, "uuid", ""),
                        "name": getattr(result, "name", ""),
                        "fact": getattr(result, "fact", ""),
                        "source_node_uuid": getattr(result.source_node, "uuid", "")
                        if hasattr(result, "source_node")
                        else "",
                        "target_node_uuid": getattr(result.target_node, "uuid", "")
                        if hasattr(result, "target_node")
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
            return SearchResult(query=query, total_count=0)

    def get_entities(
        self,
        graph_id: str,
        entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """
        Obtener entidades del grafo Graphiti

        Utiliza graphiti.nodes.get_by_group_ids()
        """
        logger.info(f"Obteniendo entidades de grafo {graph_id}...")

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Obtener nodos por group_id (graph_id)
            nodes = _run_async(graphiti.nodes.get_by_group_ids([graph_id]))

            entities = []
            for node in nodes:
                labels = getattr(node, "labels", [])
                custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

                if not custom_labels:
                    continue

                # Filtrar por tipos de entidad específicos si se proporciona
                if entity_types:
                    matching_labels = [l for l in custom_labels if l in entity_types]
                    if not matching_labels:
                        continue
                    entity_type = matching_labels[0]
                else:
                    entity_type = custom_labels[0]

                entity = EntityNode(
                    uuid=getattr(node, "uuid", ""),
                    name=getattr(node, "name", ""),
                    labels=labels,
                    summary=getattr(node, "summary", ""),
                    attributes=getattr(node, "attributes", {}),
                )

                # Enriquecer con bordes si se solicita
                if enrich_with_edges:
                    related_edges = []
                    related_node_uuids = set()

                    # Obtener bordes relacionados
                    edges = _run_async(
                        graphiti.edges.get_by_source_node_uuid(
                            getattr(node, "uuid", "")
                        )
                    )

                    for edge in edges:
                        target_uuid = (
                            getattr(edge.target_node, "uuid", "")
                            if hasattr(edge, "target_node")
                            else ""
                        )
                        if target_uuid:
                            related_edges.append(
                                {
                                    "direction": "outgoing",
                                    "edge_name": getattr(edge, "name", ""),
                                    "fact": getattr(edge, "fact", ""),
                                    "target_node_uuid": target_uuid,
                                }
                            )
                            related_node_uuids.add(target_uuid)

                    entity.related_edges = related_edges

                    # Obtener nodos relacionados
                    related_nodes = []
                    for uuid in related_node_uuids:
                        related_node = _run_async(graphiti.nodes.get(uuid))
                        if related_node:
                            related_nodes.append(
                                {
                                    "uuid": getattr(related_node, "uuid", ""),
                                    "name": getattr(related_node, "name", ""),
                                    "labels": getattr(related_node, "labels", []),
                                    "summary": getattr(related_node, "summary", ""),
                                }
                            )

                    entity.related_nodes = related_nodes

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
        Obtener una entidad por UUID en Graphiti
        """
        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            node = _run_async(graphiti.nodes.get(uuid))
            if not node:
                return None

            # Obtener bordes relacionados
            edges = _run_async(graphiti.edges.get_by_source_node_uuid(uuid))

            related_edges = []
            related_node_uuids = set()

            for edge in edges:
                target_uuid = (
                    getattr(edge.target_node, "uuid", "")
                    if hasattr(edge, "target_node")
                    else ""
                )
                if target_uuid:
                    related_edges.append(
                        {
                            "direction": "outgoing",
                            "edge_name": getattr(edge, "name", ""),
                            "fact": getattr(edge, "fact", ""),
                            "target_node_uuid": target_uuid,
                        }
                    )
                    related_node_uuids.add(target_uuid)

            # Obtener nodos relacionados
            related_nodes = []
            for related_uuid in related_node_uuids:
                related_node = _run_async(graphiti.nodes.get(related_uuid))
                if related_node:
                    related_nodes.append(
                        {
                            "uuid": getattr(related_node, "uuid", ""),
                            "name": getattr(related_node, "name", ""),
                            "labels": getattr(related_node, "labels", []),
                            "summary": getattr(related_node, "summary", ""),
                        }
                    )

            return EntityNode(
                uuid=getattr(node, "uuid", ""),
                name=getattr(node, "name", ""),
                labels=getattr(node, "labels", []),
                summary=getattr(node, "summary", ""),
                attributes=getattr(node, "attributes", {}),
                related_edges=related_edges,
                related_nodes=related_nodes,
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
        Obtener bordes del grafo Graphiti

        Graphiti usa edges.get_by_group_id() o edges.get_by_source_node_uuid()
        """
        logger.info(f"Obteniendo bordes de grafo {graph_id}...")

        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            edges_data = []

            if entity_uuid:
                # Obtener bordes por UUID de entidad
                edges = _run_async(graphiti.edges.get_by_source_node_uuid(entity_uuid))
            else:
                # Obtener todos los bordes del grupo (grafo)
                edges = _run_async(graphiti.edges.get_by_group_id(graph_id))

            for edge in edges:
                edge_dict = {
                    "uuid": getattr(edge, "uuid", ""),
                    "name": getattr(edge, "name", ""),
                    "fact": getattr(edge, "fact", ""),
                    "source_node_uuid": getattr(edge.source_node, "uuid", "")
                    if hasattr(edge, "source_node")
                    else "",
                    "target_node_uuid": getattr(edge.target_node, "uuid", "")
                    if hasattr(edge, "target_node")
                    else "",
                    "attributes": getattr(edge, "attributes", {}),
                }

                # Graphiti no tiene información temporal explícita como Zep
                if include_temporal:
                    edge_dict["created_at"] = getattr(edge, "created_at", None)
                    edge_dict["valid_at"] = getattr(edge, "valid_at", None)
                    edge_dict["invalid_at"] = getattr(edge, "invalid_at", None)

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

        Utiliza graphiti.add_episode()
        Graphiti usa group_id en lugar de graph_id
        """
        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Graphiti usa group_id para asociar episodios con un grafo
            episode_name = name or f"Episode_{reference_time or 'now'}"

            result = _run_async(
                graphiti.add_episode(
                    name=episode_name,
                    episode_body=content,
                    source=source_type,
                    reference_time=reference_time,
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

        Graphiti no tiene creación explícita de grafos
        Usa group_id en el primer episodio para crear el grafo implícitamente
        """
        import uuid

        # Generar ID del grafo
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"

        # Graphiti crea el grafo implícitamente cuando se agrega el primer episodio
        # Aquí solo devolvemos el ID
        logger.info(f"Grafo Graphiti preparado: {graph_id}")
        return graph_id

    def delete_graph(self, graph_id: str) -> bool:
        """
        Eliminar grafo Graphiti

        Graphiti usa delete_by_group_id()
        """
        try:
            self._ensure_indices()
            graphiti = self._get_graphiti()

            # Eliminar nodos del grupo
            _run_async(graphiti.nodes.delete_by_group_id(graph_id))

            logger.info(f"Grafo eliminado: {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error al eliminar grafo {graph_id}: {str(e)}")
            return False

    def build_indices(self) -> bool:
        """
        Construir índices de Neo4j para Graphiti

        Este método es crítico para Graphiti
        """
        try:
            self._ensure_indices()
            return True
        except Exception as e:
            logger.error(f"Error al construir índices: {str(e)}")
            return False
