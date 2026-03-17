"""
Servicio de herramientas de busqueda Zep
Encapsula busqueda en grafo, lectura de nodos, consulta de aristas, etc., para uso del Report agente

Herramientas de busqueda principales (optimizadas):
1. InsightForge (Busqueda de analisis profundo) - Busqueda hibrida mas potente, genera sub-preguntas automaticamente y busca en multiples dimensiones
2. PanoramaSearch (Busqueda panoramica) - Obtiene vision completa, incluyendo contenido expirado
3. QuickSearch (Busqueda rapida) - Busqueda rapida
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

# Intentar importar el sistema de prompts i18n (fallback a strings vacios si no existe)
try:
    from ..prompts import load_prompt as _load_prompt

    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False
    _load_prompt = None

logger = get_logger("mirofish.zep_tools")


@dataclass
class SearchResult:
    """Resultado de busqueda"""

    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        """Convertir a formato de texto, para comprension del LLM"""
        text_parts = [
            f"Consulta de busqueda: {self.query}",
            f"Se encontraron {self.total_count} elementos relacionados",
        ]

        if self.facts:
            text_parts.append("\n### Hechos relacionados:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Informacion del nodo"""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
        }

    def to_text(self) -> str:
        """Convertir a formato de texto"""
        entity_type = next(
            (l for l in self.labels if l not in ["Entity", "Node"]), "Tipo desconocido"
        )
        return f"Entidad: {self.name} (Tipo: {entity_type})\nResumen: {self.summary}"


@dataclass
class EdgeInfo:
    """Informacion de arista"""

    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Informacion temporal
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
        }

    def to_text(self, include_temporal: bool = False) -> str:
        """Convertir a formato de texto"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = (
            f"Relacion: {source} --[{self.name}]--> {target}\nHecho: {self.fact}"
        )

        if include_temporal:
            valid_at = self.valid_at or "Desconocido"
            invalid_at = self.invalid_at or "Hasta ahora"
            base_text += f"\nValidez: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Expirado: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """Ha expirado?"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """Es invalido?"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Resultado de busqueda de analisis profundo (InsightForge)
    Contiene resultados de multiples sub-preguntas y analisis integrado
    """

    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Resultados de busqueda por dimension
    semantic_facts: List[str] = field(
        default_factory=list
    )  # Resultados de busqueda semantica
    entity_insights: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Perspectivas de entidad
    relationship_chains: List[str] = field(default_factory=list)  # Cadenas de relacion

    # Estadisticas
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def to_text(self) -> str:
        """Convertir a formato de texto detallado, para comprension del LLM"""
        text_parts = [
            f"## Analisis de prediccion futura",
            f"Problema de analisis: {self.query}",
            f"Escenario de prediccion: {self.simulation_requirement}",
            f"\n### Estadisticas de datos de prediccion",
            f"- Hechos de prediccion relacionados: {self.total_facts}",
            f"- Entidades involucradas: {self.total_entities}",
            f"- Cadenas de relacion: {self.total_relationships}",
        ]

        # Sub-preguntas
        if self.sub_queries:
            text_parts.append(f"\n### Sub-preguntas analizadas")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        # Resultados de busqueda semantica
        if self.semantic_facts:
            text_parts.append(
                f"\n### 【Hechos clave】(Por favor cite estos en el informe)"
            )
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Perspectivas de entidad
        if self.entity_insights:
            text_parts.append(f"\n### 【Entidades principales】")
            for entity in self.entity_insights:
                text_parts.append(
                    f"- **{entity.get('name', 'Desconocido')}** ({entity.get('type', 'entidad')})"
                )
                if entity.get("summary"):
                    text_parts.append(f'  Resumen: "{entity.get("summary")}"')
                if entity.get("related_facts"):
                    text_parts.append(
                        f"  Hechos relacionados: {len(entity.get('related_facts', []))}"
                    )

        # Cadenas de relacion
        if self.relationship_chains:
            text_parts.append(f"\n### 【Cadenas de relacion】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Resultado de busqueda panoramica (Panorama)
    Contiene toda la informacion relacionada, incluyendo contenido expirado
    """

    query: str

    # Todos los nodos
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # Todas las aristas (incluyendo las expiradas)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Hechos actualmente validos
    active_facts: List[str] = field(default_factory=list)
    # Hechos expirados/invalidos (registro historico)
    historical_facts: List[str] = field(default_factory=list)

    # Estadisticas
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count,
        }

    def to_text(self) -> str:
        """Convertir a formato de texto completo (sin truncar)"""
        text_parts = [
            f"## Resultado de busqueda panoramica (Vista futura completa)",
            f"Consulta: {self.query}",
            f"\n### Estadisticas",
            f"- Total de nodos: {self.total_nodes}",
            f"- Total de aristas: {self.total_edges}",
            f"- Hechos validos actuales: {self.active_count}",
            f"- Hechos historicos/expirados: {self.historical_count}",
        ]

        # Hechos actualmente validos (salida completa, sin truncar)
        if self.active_facts:
            text_parts.append(
                f"\n### 【Hechos validos actuales】(Resultado original de simulacion)"
            )
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Hechos historicos/expirados (salida completa, sin truncar)
        if self.historical_facts:
            text_parts.append(
                f"\n### 【Hechos historicos/expirados】(Registro de evolucion)"
            )
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Entidades clave (salida completa, sin truncar)
        if self.all_nodes:
            text_parts.append(f"\n### 【Entidades involucradas】")
            for node in self.all_nodes:
                entity_type = next(
                    (l for l in node.labels if l not in ["Entity", "Node"]), "entidad"
                )
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class agenteInterview:
    """Resultado de entrevista de un agente"""

    agent_name: str
    agent_role: str  # Tipo de rol (ej: estudiante, docente, medio, etc.)
    agent_bio: str  # Biografia
    question: str  # Pregunta de entrevista
    response: str  # Respuesta de entrevista
    key_quotes: List[str] = field(default_factory=list)  # Citas clave

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes,
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Mostrar bio completa del agent, sin truncar
        text += f"_Biografia: {self.agent_bio}_\n\n"
        text += f"**P:** {self.question}\n\n"
        text += f"**R:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Citas clave:**\n"
            for quote in self.key_quotes:
                # Limpiar varios tipos de comillas
                clean_quote = (
                    quote.replace("\u201c", "").replace("\u201d", "").replace('"', "")
                )
                clean_quote = clean_quote.replace("\u300c", "").replace("\u300d", "")
                clean_quote = clean_quote.strip()
                # Eliminar signos de puntuacion al inicio
                while clean_quote and clean_quote[0] in " ,.;:!?\n\r\t ":
                    clean_quote = clean_quote[1:]
                # Filtrar contenido basura que contenga numeros de pregunta (pregunta1-9)
                skip = False
                for d in "123456789":
                    if f"question{d}" in clean_quote.lower():
                        skip = True
                        break
                if skip:
                    continue
                # Truncar contenido muy largo (por punto, no por corte duro)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find(".", 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[: dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Resultado de entrevista (Interview)
    Contiene respuestas de entrevista de multiples agentes de Simulacion
    """

    interview_topic: str  # Tema de entrevista
    interview_questions: List[str]  # Lista de preguntas de entrevista

    # Agentes seleccionados para entrevista
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Respuestas de entrevista de cada agente
    interviews: List[agenteInterview] = field(default_factory=list)

    # Razon de seleccion de agentes
    selection_reasoning: str = ""
    # Resumen de entrevista integrado
    summary: str = ""

    # Estadisticas
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count,
        }

    def to_text(self) -> str:
        """Convertir a formato de texto detallado, para comprension del LLM y citation en informes"""
        text_parts = [
            "## Informe de entrevista profunda",
            f"**Tema de entrevista:** {self.interview_topic}",
            f"**Numero de entrevista:** {self.interviewed_count} / {self.total_agents} agentes de Simulacion",
            "\n### Razon de seleccion de objetos de entrevista",
            self.selection_reasoning or "(Seleccion automatica)",
            "\n---",
            "\n### Grabacion de entrevista",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Entrevista #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(Sin registro de entrevista)\n\n---")

        text_parts.append("\n### Resumen de entrevista y puntos clave")
        text_parts.append(self.summary or "(Sin resumen)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Servicio de herramientas de busqueda Zep

    【Herramientas de busqueda principales - optimizadas】
    1. insight_forge - Busqueda de analisis profundo (la mas potente, genera sub-preguntas automaticamente, busqueda multi-dimensional)
    2. panorama_search - Busqueda panoramica (obtener vision completa, incluyendo contenido expirado)
    3. quick_search - Busqueda simple (busqueda rapida)
    4. interview_agents - Entrevista profunda (entrevistar agentes de Simulacion, obtener perspectivas multiples)

    【Herramientas basicas】
    - search_graph - Busqueda semantica del grafo
    - get_all_nodes - Obtener todos los nodos del grafo
    - get_all_edges - Obtener todas las aristas del grafo (con informacion temporal)
    - get_node_detail - Obtener informacion detallada del nodo
    - get_node_edges - Obtener aristas relacionadas del nodo
    - get_entities_by_type - Obtener entidades por tipo
    - get_entity_summary - Obtener resumen de relaciones de la entidad
    """

    # Configuracion de reintento
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(
        self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None
    ):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY no configurada")

        self.client = Zep(api_key=self.api_key)
        # Cliente LLM para generar sub-preguntas en InsightForge
        self._llm_client = llm_client
        logger.info("ZepToolsService inicializado")

    @property
    def llm(self) -> LLMClient:
        """Inicializacion perezosa del cliente LLM"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """Llamada API con mecanismo de reintento"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY

        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} intento {attempt + 1} fallo: {str(e)[:100]}, "
                        f"reintentando en {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"Zep {operation_name} fallo despues de {max_retretries} intentos: {str(e)}"
                    )

        raise last_exception

    def search_graph(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        """
        Busqueda semantica del grafo

        Utiliza busqueda hibrida (semantica + BM25) para buscar informacion relacionada en el grafo.
        Si la API de search de Zep Cloud no esta disponible, se degrada a busqueda local por palabras clave.

        Args:
            graph_id: ID del grafo (Standalone Graph)
            query: Consulta de busqueda
            limit: Numero de resultados a devolver
            scope: Alcance de busqueda, "edges" o "nodes"

        Returns:
            SearchResult: Resultado de busqueda
        """
        logger.info(f"Busqueda en grafo: graph_id={graph_id}, query={query[:50]}...")

        # Intentar usar Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder",
                ),
                operation_name=f"Busqueda en grafo(graph={graph_id})",
            )

            facts = []
            edges = []
            nodes = []

            # Parsear resultado de busqueda de aristas
            if hasattr(search_results, "edges") and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, "fact") and edge.fact:
                        facts.append(edge.fact)
                    edges.append(
                        {
                            "uuid": getattr(edge, "uuid_", None)
                            or getattr(edge, "uuid", ""),
                            "name": getattr(edge, "name", ""),
                            "fact": getattr(edge, "fact", ""),
                            "source_node_uuid": getattr(edge, "source_node_uuid", ""),
                            "target_node_uuid": getattr(edge, "target_node_uuid", ""),
                        }
                    )

            # Parsear resultado de busqueda de nodos
            if hasattr(search_results, "nodes") and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append(
                        {
                            "uuid": getattr(node, "uuid_", None)
                            or getattr(node, "uuid", ""),
                            "name": getattr(node, "name", ""),
                            "labels": getattr(node, "labels", []),
                            "summary": getattr(node, "summary", ""),
                        }
                    )
                    # El resumen del nodo tambien cuenta como hecho
                    if hasattr(node, "summary") and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(
                f"Busqueda completada: se encontraron {len(facts)} hechos relacionados"
            )

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )

        except Exception as e:
            logger.warning(
                f"Zep Search API fallo, degradando a busqueda local: {str(e)}"
            )
            # Degradacion: usar busqueda local por palabras clave
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        """
        Busqueda local por palabras clave (como plan B de Zep Search API)

        Obtiene todas las aristas/nodos y luego hace coincidir palabras clave localmente

        Args:
            graph_id: ID del grafo
            query: Consulta de busqueda
            limit: Numero de resultados a devolver
            scope: Alcance de busqueda

        Returns:
            SearchResult: Resultado de busqueda
        """
        logger.info(f"Usando busqueda local: query={query[:30]}...")

        facts = []
        edges_result = []
        nodes_result = []

        # Extraer palabras clave de la consulta (tokenizacion simple)
        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace(".", " ").split()
            if len(w.strip()) > 1
        ]

        def match_score(text: str) -> int:
            """Calcular puntuacion de coincidencia con la consulta"""
            if not text:
                return 0
            text_lower = text.lower()
            # Coincidencia exacta con la consulta
            if query_lower in text_lower:
                return 100
            # Coincidencia de palabras clave
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:
                # Obtener todas las aristas y coincidir
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))

                # Ordenar por puntuacion
                scored_edges.sort(key=lambda x: x[0], reverse=True)

                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append(
                        {
                            "uuid": edge.uuid,
                            "name": edge.name,
                            "fact": edge.fact,
                            "source_node_uuid": edge.source_node_uuid,
                            "target_node_uuid": edge.target_node_uuid,
                        }
                    )

            if scope in ["nodes", "both"]:
                # Obtener todos los nodos y coincidir
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))

                scored_nodes.sort(key=lambda x: x[0], reverse=True)

                for score, node in scored_nodes[:limit]:
                    nodes_result.append(
                        {
                            "uuid": node.uuid,
                            "name": node.name,
                            "labels": node.labels,
                            "summary": node.summary,
                        }
                    )
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(
                f"Busqueda local completada: se encontraron {len(facts)} hechos relacionados"
            )

        except Exception as e:
            logger.error(f"Busqueda local fallo: {str(e)}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts),
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Obtener todos los nodos del grafo (con paginacion)

        Args:
            graph_id: ID del grafo

        Returns:
            Lista de nodos
        """
        logger.info(f"Obtener todos los nodos del grafo {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = (
                getattr(node, "uuid_", None) or getattr(node, "uuid", None) or ""
            )
            result.append(
                NodeInfo(
                    uuid=str(node_uuid) if node_uuid else "",
                    name=node.name or "",
                    labels=node.labels or [],
                    summary=node.summary or "",
                    attributes=node.attributes or {},
                )
            )

        logger.info(f"Se obtuvo {len(result)} nodos")
        return result

    def get_all_edges(
        self, graph_id: str, include_temporal: bool = True
    ) -> List[EdgeInfo]:
        """
        Obtener todas las aristas del grafo (con paginacion, incluyendo informacion temporal)

        Args:
            graph_id: ID del grafo
            include_temporal: Si incluir informacion temporal (por defecto True)

        Returns:
            Lista de aristas (incluyendo created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(f"Obtener todas las aristas del grafo {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = (
                getattr(edge, "uuid_", None) or getattr(edge, "uuid", None) or ""
            )
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or "",
            )

            # Agregar informacion temporal
            if include_temporal:
                edge_info.created_at = getattr(edge, "created_at", None)
                edge_info.valid_at = getattr(edge, "valid_at", None)
                edge_info.invalid_at = getattr(edge, "invalid_at", None)
                edge_info.expired_at = getattr(edge, "expired_at", None)

            result.append(edge_info)

        logger.info(f"Se obtuvo {len(result)} aristas")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Obtener informacion detallada de un solo nodo

        Args:
            node_uuid: UUID del nodo

        Returns:
            Informacion del nodo o None
        """
        logger.info(f"Obtener detalle del nodo: {node_uuid[:8]}...")

        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"Obtener detalle del nodo(uuid={node_uuid[:8]}...)",
            )

            if not node:
                return None

            return NodeInfo(
                uuid=getattr(node, "uuid_", None) or getattr(node, "uuid", ""),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
            )
        except Exception as e:
            logger.error(f"Error al obtener detalle del nodo: {str(e)}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Obtener todas las aristas relacionadas con un nodo

        Obtiene todas las aristas del grafo y luego filtra las relacionadas con el nodo especificado

        Args:
            graph_id: ID del grafo
            node_uuid: UUID del nodo

        Returns:
            Lista de aristas
        """
        logger.info(f"Obtener aristas relacionadas del nodo {node_uuid[:8]}...")

        try:
            # Obtener todas las aristas del grafo y luego filtrar
            all_edges = self.get_all_edges(graph_id)

            result = []
            for edge in all_edges:
                # Verificar si la arista esta relacionada con el nodo especificado (como origen o destino)
                if (
                    edge.source_node_uuid == node_uuid
                    or edge.target_node_uuid == node_uuid
                ):
                    result.append(edge)

            logger.info(
                f"Se encontraron {len(result)} aristas relacionadas con el nodo"
            )
            return result

        except Exception as e:
            logger.warning(f"Error al obtener aristas del nodo: {str(e)}")
            return []

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """
        Obtener entidades por tipo

        Args:
            graph_id: ID del grafo
            entity_type: Tipo de entidad (ej Student, PublicFigure, etc.)

        Returns:
            Lista de entidades que coinciden con el tipo
        """
        logger.info(f"Obtener entidades de tipo {entity_type}...")

        all_nodes = self.get_all_nodes(graph_id)

        filtered = []
        for node in all_nodes:
            # Verificar si las labels incluyen el tipo especificado
            if entity_type in node.labels:
                filtered.append(node)

        logger.info(f"Se encontraron {len(filtered)} entidades de tipo {entity_type}")
        return filtered

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """
        Obtener resumen de relaciones de una entidad especifica

        Busca toda la informacion relacionada con esa entidad y genera un resumen

        Args:
            graph_id: ID del grafo
            entity_name: Nombre de la entidad

        Returns:
            Informacion de resumen de la entidad
        """
        logger.info(f"Obtener resumen de relaciones de la entidad {entity_name}...")

        # Primero buscar informacion relacionada con esa entidad
        search_result = self.search_graph(
            graph_id=graph_id, query=entity_name, limit=20
        )

        # Intentar encontrar esa entidad en todos los nodos
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            # Pasar parametro graph_id
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Obtener estadisticas del grafo

        Args:
            graph_id: ID del grafo

        Returns:
            Informacion estadistica
        """
        logger.info(f"Obtener estadisticas del grafo {graph_id}...")

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        # Contar distribucion de tipos de entidad
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        # Contar distribucion de tipos de relacion
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1

        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def get_simulation_context(
        self, graph_id: str, simulation_requirement: str, limit: int = 30
    ) -> Dict[str, Any]:
        """
        Obtener informacion de contexto relacionada con la Simulacion

        Busqueda integral de toda la informacion relacionada con el requisito de Simulacion

        Args:
            graph_id: ID del grafo
            simulation_requirement: Descripcion del requisito de Simulacion
            limit: Limite de cantidad de informacion de cada tipo

        Returns:
            Informacion de contexto de Simulacion
        """
        logger.info(f"Obtener contexto de Simulacion: {simulation_requirement[:50]}...")

        # Buscar informacion relacionada con el requisito de Simulacion
        search_result = self.search_graph(
            graph_id=graph_id, query=simulation_requirement, limit=limit
        )

        # Obtener estadisticas del grafo
        stats = self.get_graph_statistics(graph_id)

        # Obtener todos los nodos de entidad
        all_nodes = self.get_all_nodes(graph_id)

        # Filtrar entidades con tipo real (no nodos Entity puros)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append(
                    {
                        "name": node.name,
                        "type": custom_labels[0],
                        "summary": node.summary,
                    }
                )

        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Limitar cantidad
            "total_entities": len(entities),
        }

    # ========== Herramientas de busqueda principales (optimizadas) ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """
        【InsightForge - Busqueda de analisis profundo】

        La funcion de busqueda hibrida mas potente, descompone automaticamente preguntas y busca en multiples dimensiones:
        1. Usar LLM para descomponer la pregunta en multiples sub-preguntas
        2. Hacer busqueda semantica para cada sub-pregunta
        3. Extraer entidades relacionadas y obtener su informacion detallada
        4. Rastrear cadenas de relacion
        5. Integrar todos los resultados, generar analisis profundo

        Args:
            graph_id: ID del grafo
            query: Pregunta del usuario
            simulation_requirement: Descripcion del requisito de Simulacion
            report_context: Contexto del informe (opcional, para generacion mas precisa de sub-preguntas)
            max_sub_queries: Numero maximo de sub-preguntas

        Returns:
            InsightForgeResult: Resultado de busqueda de analisis profundo
        """
        logger.info(f"InsightForge busqueda de analisis profundo: {query[:50]}...")

        result = InsightForgeResult(
            query=query, simulation_requirement=simulation_requirement, sub_queries=[]
        )

        # Step 1: Usar LLM para generar sub-preguntas
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info(f"Generar {len(sub_queries)} sub-preguntas")

        # Step 2: Hacer busqueda semantica para cada sub-pregunta
        all_facts = []
        all_edges = []
        seen_facts = set()

        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id, query=sub_query, limit=15, scope="edges"
            )

            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)

            all_edges.extend(search_result.edges)

        # Tambien buscar con la pregunta original
        main_search = self.search_graph(
            graph_id=graph_id, query=query, limit=20, scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Step 3: Extraer UUIDs de entidades relacionadas de las aristas, solo obtener informacion de esas entidades (no de todos los nodos)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get("source_node_uuid", "")
                target_uuid = edge_data.get("target_node_uuid", "")
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        # Obtener detalles de todas las entidades relacionadas (sin limite, salida completa)
        entity_insights = []
        node_map = {}  # Para construccion posterior de cadenas de relacion

        for uuid in list(entity_uuids):  # Procesar todas las entidades, sin truncar
            if not uuid:
                continue
            try:
                # Obtener informacion de cada nodo relacionado individualmente
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next(
                        (l for l in node.labels if l not in ["Entity", "Node"]),
                        "entidad",
                    )

                    # Obtener todos los hechos relacionados con esa entidad (sin truncar)
                    related_facts = [
                        f for f in all_facts if node.name.lower() in f.lower()
                    ]

                    entity_insights.append(
                        {
                            "uuid": node.uuid,
                            "name": node.name,
                            "type": entity_type,
                            "summary": node.summary,
                            "related_facts": related_facts,  # Salida completa, sin truncar
                        }
                    )
            except Exception as e:
                logger.debug(f"Error al obtener nodo {uuid}: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Step 4: Construir todas las cadenas de relacion (sin limite)
        relationship_chains = []
        for edge_data in all_edges:  # Procesar todas las aristas, sin truncar
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get("source_node_uuid", "")
                target_uuid = edge_data.get("target_node_uuid", "")
                relation_name = edge_data.get("name", "")

                source_name = (
                    node_map.get(source_uuid, NodeInfo("", "", [], "", {})).name
                    or source_uuid[:8]
                )
                target_name = (
                    node_map.get(target_uuid, NodeInfo("", "", [], "", {})).name
                    or target_uuid[:8]
                )

                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(
            f"InsightForge completado: {result.total_facts} hechos, {result.total_entities} entidades, {result.total_relationships} relaciones"
        )
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        """
        Usar LLM para generar sub-preguntas

        Descomponer una pregunta compleja en multiples sub-preguntas que se pueden buscar independientemente
        """
        # Usar prompts i18n si estan disponibles
        if _PROMPTS_AVAILABLE:
            system_prompt = _load_prompt("zep", "sub_query_generation")
        else:
            # Fallback al original en chino
            system_prompt = """Eres un experto en analisis de preguntas. Tu tarea es descomponer una pregunta compleja en multiples sub-preguntas que se pueden observar independientemente en el mundo de Simulacion.

Requisitos:
1. Cada sub-pregunta debe ser lo suficientemente especifica para encontrar comportamientos o eventos de agentes relacionados en el mundo de Simulacion
2. Las sub-preguntas deben cubrir diferentes dimensiones de la pregunta original (como: quien, que, por que, como, cuando, donde)
3. Las sub-preguntas deben estar relacionadas con el escenario de Simulacion
4. Devolver en formato JSON: {"sub_queries": ["sub-pregunta1", "sub-pregunta2", ...]}"""

        user_prompt = f"""Contexto del requisito de Simulacion:
{simulation_requirement}

{f"Contexto del informe: {report_context[:500]}" if report_context else ""}

Por favor descomponer la siguiente pregunta en {max_queries} sub-preguntas:
{query}

Devolver lista de sub-preguntas en formato JSON."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            sub_queries = response.get("sub_queries", [])
            # Asegurar que es una lista de strings
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(
                f"Error al generar sub-preguntas: {str(e)}, usando sub-preguntas por defecto"
            )
            # Degradacion: devolver variantes basadas en la pregunta original
            return [
                query,
                f"Participantes principales de {query}",
                f"Causas y efectos de {query}",
                f"Proceso de desarrollo de {query}",
            ][:max_queries]

    def panorama_search(
        self, graph_id: str, query: str, include_expired: bool = True, limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch - Busqueda panoramica】

        Obtener vista completa, incluyendo todo el contenido relacionado e historico/expirado:
        1. Obtener todos los nodos relacionados
        2. Obtener todas las aristas (incluyendo las expiradas/invalidas)
        3. Clasificar y organizar informacion valida actual e historica

        Esta herramienta es adecuada para escenarios que requieren conocer la vision completa de un evento y rastrear el proceso de evolucion.

        Args:
            graph_id: ID del grafo
            query: Consulta de busqueda (para ordenamiento por relevancia)
            include_expired: Si incluir contenido expirado (por defecto True)
            limit: Limite de cantidad de resultados a devolver

        Returns:
            PanoramaResult: Resultado de busqueda panoramica
        """
        logger.info(f"PanoramaSearch busqueda panoramica: {query[:50]}...")

        result = PanoramaResult(query=query)

        # Obtener todos los nodos
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # Obtener todas las aristas (incluyendo informacion temporal)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Clasificar hechos
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            # Agregar nombres de entidad a los hechos
            source_name = (
                node_map.get(edge.source_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.source_node_uuid[:8]
            )
            target_name = (
                node_map.get(edge.target_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.target_node_uuid[:8]
            )

            # Determinar si esta expirado/invalido
            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                # Hecho historico/expirado, agregar marca de tiempo
                valid_at = edge.valid_at or "Desconocido"
                invalid_at = edge.invalid_at or edge.expired_at or "Desconocido"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Hecho valido actual
                active_facts.append(edge.fact)

        # Ordenar por relevancia basado en la consulta
        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace(".", " ").split()
            if len(w.strip()) > 1
        ]

        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score

        # Ordenar y limitar cantidad
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(
            f"PanoramaSearch completado: {result.active_count} validos, {result.historical_count} historicos"
        )
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """
        【QuickSearch - Busqueda simple】

        Herramienta de busqueda rapida y ligera:
        1. Llamar directamente a busqueda semantica de Zep
        2. Devolver los resultados mas relevantes
        3. Apropiado para requisitos de busqueda simples y directos

        Args:
            graph_id: ID del grafo
            query: Consulta de busqueda
            limit: Numero de resultados a devolver

        Returns:
            SearchResult: Resultado de busqueda
        """
        logger.info(f"QuickSearch busqueda simple: {query[:50]}...")

        # Llamar directamente al metodo search_graph existente
        result = self.search_graph(
            graph_id=graph_id, query=query, limit=limit, scope="edges"
        )

        logger.info(f"QuickSearch completado: {result.total_count} resultados")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        """
        【InterviewAgents - Entrevista profunda】

        Llama a la API real de entrevista de OASIS, entrevista agentes que se estan ejecutando en la Simulacion:
        1. Leer automaticamente archivos de perfil, conocer todos los agentes de Simulacion
        2. Usar LLM para analizar requisito de entrevista, seleccionar agentes mas relevantes inteligentemente
        3. Usar LLM para generar preguntas de entrevista
        4. Llamar interfaz /api/simulation/interview/batch para entrevista real (ambas plataformas entrevistando simultaneamente)
        5. Integrar todos los resultados de entrevista, generar informe de entrevista

        【Importante】Esta funcion requiere que el entorno de Simulacion este en ejecucion (entorno OASIS no cerrado)

        【Escenarios de uso】
        - Se necesita conocer opiniones de diferentes roles sobre eventos
        - Se necesita recolectar opiniones y puntos de vista de multiples partes
        - Se necesita obtener respuestas reales de agentes de Simulacion (no Simulacion LLM)

        Args:
            simulation_id: ID de Simulacion (para localizar archivos de perfil y llamar API de entrevista)
            interview_requirement: Descripcion del requisito de entrevista (no estructurado, como "conocer opiniones de estudiantes sobre el evento")
            simulation_requirement: Contexto del requisito de Simulacion (opcional)
            max_agents: Numero maximo de agentes a entrevistar
            custom_questions: Preguntas de entrevista personalizadas (opcional, si no se proporcionan se generan automaticamente)

        Returns:
            InterviewResult: Resultado de entrevista
        """
        from .simulation_runner import SimulationRunner

        logger.info(
            f"InterviewAgents entrevista profunda (API real): {interview_requirement[:50]}..."
        )

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        # Step 1: Leer archivos de perfil
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(
                f"No se encontraron archivos de perfil de agentes para Simulacion {simulation_id}"
            )
            result.summary = (
                "No se encontraron archivos de perfil de agentes para entrevistar"
            )
            return result

        result.total_agents = len(profiles)
        logger.info(f"Se cargaron {len(profiles)} perfiles de agentes")

        # Step 2: Usar LLM para seleccionar agentes a entrevistar (devolver lista de agent_id)
        selected_agents, selected_indices, selection_reasoning = (
            self._select_agents_for_interview(
                profiles=profiles,
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                max_agents=max_agents,
            )
        )

        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(
            f"Se seleccionaron {len(selected_agents)} agentes para entrevista: {selected_indices}"
        )

        # Step 3: Generar preguntas de entrevista (si no se proporcionan)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )
            logger.info(
                f"Se generaron {len(result.interview_questions)} preguntas de entrevista"
            )

        # Combinar preguntas en un prompt de entrevista
        combined_prompt = "\n".join(
            [f"{i + 1}. {q}" for i, q in enumerate(result.interview_questions)]
        )

        # Agregar prefijo optimizado para restringir formato de respuesta del agente (i18n)
        if _PROMPTS_AVAILABLE:
            INTERVIEW_PROMPT_PREFIX = _load_prompt("zep", "interview_prompt_prefix")
        else:
            # Fallback al original en chino
            INTERVIEW_PROMPT_PREFIX = (
                "Estas en una entrevista. Por favor combina tu perfil, todos tus recuerdos y acciones pasados, "
                "y responde directamente las siguientes preguntas en texto plano.\n"
                "Requisitos de respuesta:\n"
                "1. Usa lenguaje natural directamente, no llames a ninguna herramienta\n"
                "2. No devuelvas formato JSON ni formato de llamada de herramienta\n"
                "3. No uses titulos Markdown (como #, ##, ###)\n"
                "4. Responde cada pregunta numerada, cada respuesta comienza con 「preguntaX:」 (X es el numero de pregunta)\n"
                "5. Separa las respuestas de cada pregunta con lineas en blanco\n"
                "6. Las respuestas deben tener contenido sustancial, al menos 2-3 oraciones por pregunta\n\n"
            )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: Llamar API real de entrevista (sin especificar platform, por defecto ambas plataformas)
        try:
            # Construir lista de entrevista por lotes (sin especificar platform, entrevista en ambas plataformas)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append(
                    {
                        "agent_id": agent_idx,
                        "prompt": optimized_prompt,  # Usar prompt optimizado
                        # No especificar platform, la API entrevistara en twitter y reddit
                    }
                )

            logger.info(
                f"Llamar API de entrevista por lotes (ambas plataformas): {len(interviews_request)} agentes"
            )

            # Llamar metodo de entrevista por lotes de SimulationRunner (sin pasar platform, entrevista en ambas plataformas)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # No especificar platform, entrevista en ambas plataformas
                timeout=180.0,  # Ambas plataformas necesitan mas tiempo de espera
            )

            logger.info(
                f"API de entrevista retorno: {api_result.get('interviews_count', 0)} resultados, success={api_result.get('success')}"
            )

            # Verificar si la llamada API fue exitosa
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Error desconocido")
                logger.warning(f"API de entrevista retorno fallo: {error_msg}")
                result.summary = f"Llamada a API de entrevista fallo: {error_msg}. Por favor verificar estado del entorno de Simulacion OASIS."
                return result

            # Step 5: Parsear resultado de API, construir objetos de agenteInterview
            # Formato de retorno en modo doble plataforma: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = (
                api_data.get("results", {}) if isinstance(api_data, dict) else {}
            )

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get(
                    "realname", agent.get("username", f"agente_{agent_idx}")
                )
                agent_role = agent.get("profession", "Desconocido")
                agent_bio = agent.get("bio", "")

                # Obtener resultado de entrevista del agente en ambas plataformas
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Limpiar posible envoltorio de llamada de herramienta JSON
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Siempre mostrar marca de doble plataforma
                twitter_text = (
                    twitter_response
                    if twitter_response
                    else "(No se obtuvo respuesta en esta plataforma)"
                )
                reddit_text = (
                    reddit_response
                    if reddit_response
                    else "(No se obtuvo respuesta en esta plataforma)"
                )
                response_text = f"【Respuesta de Twitter】\n{twitter_text}\n\n【Respuesta de Reddit】\n{reddit_text}"

                # Extraer citas clave (de las respuestas de ambas plataformas)
                import re

                combined_responses = f"{twitter_response} {reddit_response}"

                # Limpiar texto de respuesta: quitar marcas, numeros, Markdown, etc.
                clean_text = re.sub(r"#{1,6}\s+", "", combined_responses)
                clean_text = re.sub(r"\{[^}]*tool_name[^}]*\}", "", clean_text)
                clean_text = re.sub(r"[*_`|>~\-]{2,}", "", clean_text)
                clean_text = re.sub(r"pregunta\d+[：:]\s*", "", clean_text)
                clean_text = re.sub(r"【[^】]+】", "", clean_text)

                # Estrategia 1 (principal): Extraer oraciones completas con contenido sustancial
                sentences = re.split(r"[.!?]", clean_text)
                meaningful = [
                    s.strip()
                    for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r"^[\s\W,.;:!?]+", s.strip())
                    and not s.strip().startswith(("{", "pregunta"))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "." for s in meaningful[:3]]

                # Estrategia 2 (complementaria): Texto largo dentro de comillas「」 correctamente emparejadas
                if not key_quotes:
                    paired = re.findall(
                        r"\u201c([^\u201c\u201d]{15,100})\u201d", clean_text
                    )
                    paired += re.findall(
                        r"\u300c([^\u300c\u300d]{15,100})\u300d", clean_text
                    )
                    key_quotes = [q for q in paired if not re.match(r"^[,.;:!?]", q)][
                        :3
                    ]

                interview = agenteInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # Ampliar limite de longitud de bio
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            # Entorno de Simulacion no esta en ejecucion
            logger.warning(
                f"Llamada a API de entrevista fallo (entorno no en ejecucion?): {e}"
            )
            result.summary = f"Entrevista fallo: {str(e)}. El entorno de Simulacion puede haber sido cerrado, por favor asegurar que el entorno OASIS este en ejecucion."
            return result
        except Exception as e:
            logger.error(f"Excepcion en llamada a API de entrevista: {e}")
            import traceback

            logger.error(traceback.format_exc())
            result.summary = f"Error en proceso de entrevista: {str(e)}"
            return result

        # Step 6: Generar resumen de entrevista
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )

        logger.info(
            f"InterviewAgents completado: se entrevistaron {result.interviewed_count} agentes (doble plataforma)"
        )
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """Limpiar envoltorio de llamada de herramienta JSON de la respuesta del agente, extraer contenido real"""
        if not response or not response.strip().startswith("{"):
            return response
        text = response.strip()
        if "tool_name" not in text[:80]:
            return response
        import re as _re

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "arguments" in data:
                for key in ("content", "text", "body", "message", "reply"):
                    if key in data["arguments"]:
                        return str(data["arguments"][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace("\\n", "\n").replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Cargar archivos de perfil de agentes de Simulacion"""
        import os
        import csv

        # Construir ruta de archivo de perfil
        sim_dir = os.path.join(
            os.path.dirname(__file__), f"../../uploads/simulations/{simulation_id}"
        )

        profiles = []

        # Prioridad: intentar leer formato Reddit JSON
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                logger.info(
                    f"Se cargaron {len(profiles)} perfiles de reddit_profiles.json"
                )
                return profiles
            except Exception as e:
                logger.warning(f"Error al leer reddit_profiles.json: {e}")

        # Intentar leer formato Twitter CSV
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convertir formato CSV a formato unificado
                        profiles.append(
                            {
                                "realname": row.get("name", ""),
                                "username": row.get("username", ""),
                                "bio": row.get("description", ""),
                                "persona": row.get("user_char", ""),
                                "profession": "Desconocido",
                            }
                        )
                logger.info(
                    f"Se cargaron {len(profiles)} perfiles de twitter_profiles.csv"
                )
                return profiles
            except Exception as e:
                logger.warning(f"Error al leer twitter_profiles.csv: {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> tuple:
        """
        Usar LLM para seleccionar agentes a entrevistar

        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: Lista de informacion completa de agentes seleccionados
                - selected_indices: Lista de indices de agentes seleccionados (para llamada API)
                - reasoning: Razon de seleccion
        """

        # Construir lista de resumen de agentes
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"agente_{i}")),
                "profession": profile.get("profession", "Desconocido"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", []),
            }
            agent_summaries.append(summary)

        system_prompt = """Eres un experto en planificacion de entrevistas. Tu tarea es seleccionar los objetos mas apropiados para entrevista de la lista de agentes de Simulacion segun el requisito de entrevista.

Criterios de seleccion:
1. La identidad/profesion del agente esta relacionada con el tema de la entrevista
2. El agente puede tener puntos de vista unicos o valiosos
3. Seleccionar perspectivas diversas (como: a favor, en contra, neutral, profesional, etc.)
4. Priorizar roles directamente relacionados con el evento

Devolver en formato JSON:
{
    "selected_indices": [lista de indices de agentes seleccionados],
    "reasoning": "Explicacion de la razon de seleccion"
}"""

        user_prompt = f"""Requisito de entrevista:
{interview_requirement}

Contexto de Simulacion:
{simulation_requirement if simulation_requirement else "No proporcionado"}

Lista de agentes disponibles (total {len(agent_summaries)}):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Por favor seleccionar hasta {max_agents} agentes mas apropiados para entrevista y explicar la razon de seleccion."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get(
                "reasoning", "Seleccion automatica basada en relevancia"
            )

            # Obtener informacion completa de los agentes seleccionados
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(
                f"Error al seleccionar agentes con LLM, usando seleccion por defecto: {e}"
            )
            # Degradacion: seleccionar los primeros N
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Usando estrategia de seleccion por defecto"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """Usar LLM para generar preguntas de entrevista"""

        agent_roles = [a.get("profession", "Desconocido") for a in selected_agents]

        system_prompt = """Eres un periodista/experto en entrevistas profesional. Segun el requisito de entrevista, generar 3-5 preguntas de entrevista profundas.

Requisitos de preguntas:
1. Preguntas abiertas que fomenten respuestas detalladas
2. Diferentes respuestas para diferentes roles
3. Cubrir multiples dimensiones como hechos, puntos de vista, sentimientos, etc.
4. Lenguaje natural, como una entrevista real
5. Cada pregunta controlada en 50 caracteres o menos, concisa y clara
6. Preguntar directamente, no incluir explicaciones de fondo o prefijos

Devolver en formato JSON: {"questions": ["pregunta1", "pregunta2", ...]}"""

        user_prompt = f"""Requisito de entrevista: {interview_requirement}

Contexto de Simulacion: {simulation_requirement if simulation_requirement else "No proporcionado"}

Roles de objetos de entrevista: {", ".join(agent_roles)}

Por favor generar 3-5 preguntas de entrevista."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )

            return response.get(
                "questions", [f"Cual es su opinion sobre {interview_requirement}?"]
            )

        except Exception as e:
            logger.warning(f"Error al generar preguntas de entrevista: {e}")
            return [
                f"Cual es su opinion sobre {interview_requirement}?",
                "Como le afecta a usted o al grupo que representa este asunto?",
                "Como cree que se deberia resolver o mejorar este tema?",
            ]

    def _generate_interview_summary(
        self, interviews: List[agenteInterview], interview_requirement: str
    ) -> str:
        """Generar resumen de entrevista"""

        if not interviews:
            return "No se completo ninguna entrevista"

        # Recopilar todo el contenido de entrevistas
        interview_texts = []
        for interview in interviews:
            interview_texts.append(
                f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}"
            )

        system_prompt = """Eres un editor de noticias profesional. Segun las respuestas de multiples entrevistados, generar un resumen de entrevista.

Requisitos del resumen:
1. Extraer los puntos de vista principales de cada parte
2. Senalar consenso y divergencia de puntos de vista
3. Destacar citas valiosas
4. Objetivo y neutral, sin favorecer a ninguna parte
5. Controlar en 1000 caracteres

Restricciones de formato (deben cumplirse):
- Usar parrafos de texto plano, separar diferentes partes con lineas en blanco
- No usar titulos Markdown (como #, ##, ###)
- No usar lineas divisorias (como ---, ***)
- Usar comillas「」al citar respuestas originales de los entrevistados
- Se puede usar **negrita** para marcar palabras clave, pero no usar otras sintaxis Markdown"""

        user_prompt = f"""Tema de entrevista: {interview_requirement}

Contenido de entrevista:
{"".join(interview_texts)}

Por favor generar un resumen de entrevista."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            return summary

        except Exception as e:
            logger.warning(f"Error al generar resumen de entrevista: {e}")
            # Degradacion: concatenacion simple
            return (
                f"Se entrevistaron {len(interviews)} entrevistados, incluyendo: "
                + ", ".join([i.agent_name for i in interviews])
            )
