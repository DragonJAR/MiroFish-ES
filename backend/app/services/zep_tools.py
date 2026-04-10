"""
Servicio de herramientas de búsqueda Zep
Encapsula herramientas de búsqueda de grafo, lectura de nodos, consulta de bordes, etc., para uso del agentee de Reportes

herramientas de búsqueda centrales (optimizadas):
1. InsightForge (búsqueda de perspicacia profunda) - búsqueda híbrida Más potente, genera sub-preguntas automáticaMente y busca en múltiples dimensiones
2. PanoramaSearch (búsqueda en amplitud) - Obtiene visión completa, incluyendo contenido expIrado
3. QuickSearch (búsqueda simple) - búsqueda rápida
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, t
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger("mirofish.zep_tools")


@dataclass
class SearchResult:
    """resultados de búsqueda"""

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
        """Convertir a Formato de texto para Comprensión del LLM"""
        text_parts = [
            f"consulta de búsqueda: {self.query}",
            f"Encontrado {self.total_count} inFormaciones relacionadas",
        ]

        if self.facts:
            text_parts.append(f"\n### {t('console.zep.relatedFacts')}:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """información del nodo"""

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
        """Convertir a Formato de texto"""
        entity_type = next(
            (l for l in self.labels if l not in ["Entity", "Node"]), "Desconocidotipo"
        )
        return f"entidad: {self.name} (tipo: {entity_type})\n{t('console.zep.summary')}: {self.summary}"


@dataclass
class EdgeInfo:
    """información del borde"""

    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Tiempoinformación
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
        """Convertir a texto"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"relación: {source} --[{self.name}]--> {target}\nreal: {self.fact}"

        if include_temporal:
            valid_at = self.valid_at or "Desconocido"
            invalid_at = self.invalid_at or t("console.zep.toDate")
            base_text += f"\n{t('console.zep.validity')}: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" ({t('console.zep.expired')}: {self.expired_at})"

        return base_text

    @property
    def is_expired(self) -> bool:
        """{t('console.zep.ifExpired')}"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """{t('console.zep.ifInvalidated')}"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    resultado de búsqueda de perspicacia profunda (InsightForge)
    Contiene resultados de búsqueda de múltiples sub-preguntas, así como análisis integral
    """

    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # resultados de búsqueda en cada dimensión
    semantic_facts: List[str] = field(
        default_factory=list
    )  # resultados de búsqueda semántica
    entity_insights: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Perspicacias de entidad
    relationship_chains: List[str] = field(default_factory=list)  # Cadena de relaciones

    # información de estadísticas
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
        """Convertir a Formato de texto detallado para Comprensión del LLM"""
        text_parts = [
            f"## Análisis Profundo de predicción futura",
            f"problema de análisis: {self.query}",
            f"Escenario de predicción: {self.simulation_requirement}",
            f"\n### Estadísticas de datos de predicción",
            f"- hechos de predicción relacionados: {self.total_facts} elementos",
            f"- entidades involucradas: {self.total_entities} elementos",
            f"- Cadena de relaciones: {self.total_relationships} elementos",
        ]

        # Sub-preguntas
        if self.sub_queries:
            text_parts.append(f"\n### Sub-preguntas del análisis")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        # resultados de búsqueda semántica
        if self.semantic_facts:
            text_parts.append(
                f"\n### 【hechos clave】(por favor cite estos textos originales en el reporte)"
            )
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # Perspicacias de entidad
        if self.entity_insights:
            text_parts.append(f"\n### 【entidades principales】")
            for entity in self.entity_insights:
                text_parts.append(
                    f"- **{entity.get('name', 'Desconocido')}** ({entity.get('type', 'entidad')})"
                )
                if entity.get("summary"):
                    text_parts.append(f'  Resumen: "{entity.get("summary")}"')
                if entity.get("reLated_facts"):
                    text_parts.append(
                        f"  hechos relacionados: {len(entity.get('reLated_facts', []))} elementos"
                    )

        # Cadena de relaciones
        if self.relationship_chains:
            text_parts.append(f"\n### 【Cadena de relaciones】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    resultados de búsqueda en amplitud (Panorama)
    Contiene toda la inFormación relevante, incluyendo contenido expIrado
    """

    query: str

    # Todos los nodos
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # Todos los bordes (incluyendo expIrados)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # hechos actualMente válidos
    active_facts: List[str] = field(default_factory=list)
    # hechos expIrados/inválidos (registros históricos)
    historical_facts: List[str] = field(default_factory=list)

    # Estadísticas
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
        """Convertir a Formato de texto (versión completa, sin truncamiento)"""
        text_parts = [
            f"## resultados de búsqueda en amplitud (vista panorámica futura)",
            f"consulta: {self.query}",
            f"\n### información de estadísticas",
            f"- Número total de nodos: {self.total_nodes}",
            f"- Número total de bordes: {self.total_edges}",
            f"- hechos actualMente válidos: {self.active_count} elementos",
            f"- hechos históricos/exprIrados: {self.historical_count} elementos",
        ]

        # hechos actualMente válidos (salida completa, sin truncamiento)
        if self.active_facts:
            text_parts.append(
                f"\n### 【hechos actualMente válidos】(texto original de resultados de simulación)"
            )
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # hechos históricos/exprIrados (salida completa, sin truncamiento)
        if self.historical_facts:
            text_parts.append(
                f"\n### 【hechos históricos/exprIrados】(registro de Proceso de evolución)"
            )
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # entidades clave (salida completa, sin truncamiento)
        if self.all_nodes:
            text_parts.append(f"\n### 【entidades involucradas】")
            for node in self.all_nodes:
                entity_type = next(
                    (l for l in node.labels if l not in ["Entity", "Node"]), "entidad"
                )
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class agenteInterview:
    """resultado de entrevista de un solo agente"""

    agent_name: str
    agent_role: str  # tipo de rol (ej: estudiante, proFesor, Medios, etc.)
    agent_bio: str  # Biografía
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
        # Mostrar bio completa del agente, sin truncamiento
        text += f"_Biografía: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Citas clave:**\n"
            for quote in self.key_quotes:
                # Limpiar varios tipos de comillas
                clean_quote = (
                    quote.replace("\u201c", "").replace("\u201d", "").replace('"', "")
                )
                clean_quote = clean_quote.replace("\u300c", "").replace("\u300d", "")
                clean_quote = clean_quote.strip()
                # Eliminar puntuación al inicio
                while clean_quote and clean_quote[0] in "，,；;：:、。！？\n\r\t ":
                    clean_quote = clean_quote[1:]
                # filtrado contenido basura que incluye números de preguntas (problema1-9）
                skip = False
                for d in "123456789":
                    if f"\u95ee\u9898{d}" in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Truncar contenido Demasiado Largo (truncar en punto en lugar de truncamiento Duro)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find("\u3002", 80)
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
    resultado de entrevista (Interview)
    Contiene Respuestas de entrevista de múltiples agentes de simulación
    """

    interview_topic: str  # Tema de entrevista
    interview_questions: List[str]  # Lista de preguntas de entrevista

    # agentees seleccionados para entrevista
    selected_agentes: List[Dict[str, Any]] = field(default_factory=list)
    # Respuestas de entrevista de cada agente
    interviews: List[agenteInterview] = field(default_factory=list)

    # Razón de selección de agente
    selection_reasoning: str = ""
    # Resumen de entrevista después de integración
    summary: str = ""

    # Estadísticas
    total_agentes: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agentes": self.selected_agentes,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agentes": self.total_agentes,
            "interviewed_count": self.interviewed_count,
        }

    def to_text(self) -> str:
        """Convertir a Formato de texto detallado para Comprensión del LLM y reFerencia en reporte"""
        text_parts = [
            "## Informe de entrevista profunda",
            f"**Tema de entrevista:** {self.interview_topic}",
            f"**Número de entrevistados:** {self.interviewed_count} / {self.total_agentes} agentes de simulación",
            "\n### Razón de selección de Objeto de entrevista",
            self.selection_reasoning or "(selección automática)",
            "\n---",
            "\n### Registro de entrevista",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Entrevista #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(Ningun registro de entrevista)\n\n---")

        text_parts.append("\n### Resumen de entrevista y puntos clave")
        text_parts.append(self.summary or "(Ningún resumen)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Servicio de herramientas de búsqueda Zep

    【herramientas de búsqueda centrales - optimizadas】
    1. insight_forge - búsqueda de perspicacia profunda (Más potente, genera sub-preguntas automáticaMente, búsqueda en múltiples dimensiones)
    2. panorama_search - búsqueda en amplitud (obtiene visión completa, incluyendo contenido expIrado)
    3. quick_search - búsqueda simple (búsqueda rápida)
    4. interview_agentes - Entrevista profunda (entrevista agentes de simulación, obtiene Perspectivas múltiples)

    【herramientas básicas】
    - search_graph - búsqueda semántica de grafo
    - get_all_nodes - obtener todos los nodos del grafo
    - get_all_edges - obtener todos los bordes del grafo (incluyendo inFormación temporal)
    - get_node_detail - obtener inFormación detallada del nodo
    - get_node_edges - obtener bordes relacionados con el nodo
    - get_entities_by_type - obtener entidades por tipo
    - get_entity_summary - obtener resumen de relaciones de la entidad
    """

    # Configuración de reintento
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(
        self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None
    ):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY no está configurado")

        self.client = Zep(api_key=self.api_key)
        # Cliente LLM para InsightForge generar sub-preguntas
        self._llm_client = llm_client
        logger.info(t("console.zepToolsInitialized"))

    @property
    def llm(self) -> LLMClient:
        """Inicialización perezosa del cliente LLM"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def _call_with_retry(self, func, opeRation_name: str, max_retries: int = None):
        """llamada a API con Mecanismo de reintento"""
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
                        t(
                            "console.zepRetryAttempt",
                            opeRation=opeRation_name,
                            attempt=attempt + 1,
                            error=str(e)[:100],
                            delay=f"{delay:.1f}",
                        )
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        t(
                            "console.zepAllRetriesFailed",
                            opeRation=opeRation_name,
                            retries=max_retries,
                            error=str(e),
                        )
                    )

        raise last_exception

    def search_graph(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        """
        búsqueda semántica de grafo

        Usa búsqueda híbrida (semántica+BM25) para buscar inFormación relevante en el grafo.
        Si la API de búsqueda de Zep Cloud no está disponible, degrada a coincidencia de palabras clave locales.

        Args:
            graph_id: ID del Grafo (Grafo independiente)
            query: consulta de búsqueda
            limit: cantidad de resultados a volver
            scope: Rango de búsqueda, "edges" o "nodes"

        Returns:
            SearchResult: resultados de búsqueda
        """
        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))

        # Intentar usar API de búsqueda Zep Cloud
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder",
                ),
                opeRation_name=t("console.graphSearchOp", graphId=graph_id),
            )

            facts = []
            edges = []
            nodes = []

            # Analizar bordes de resultados de búsqueda
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

            # Analizar nodos de resultados de búsqueda
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
                    # Resumen de nodo también cuenta como hecho
                    if hasattr(node, "summary") and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(t("console.searchComplete", count=len(facts)))

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )

        except Exception as e:
            logger.warning(t("console.zepSearchApiFallback", error=str(e)))
            # Degradar: usar coincidencia de palabras clave locales
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        """
        búsqueda de coincidencia de palabras clave local (como solución Alternativa de la API de búsqueda Zep)

        obtener todos los bordes/nodos, luego realizar coincidencia de palabras clave localMente

        Args:
            graph_id: ID del Grafo
            query: consulta de búsqueda
            limit: cantidad de resultados a volver
            scope: Rango de búsqueda

        Returns:
            SearchResult: resultados de búsqueda
        """
        logger.info(t("console.usingLocalSearch", query=query[:30]))

        facts = []
        edges_result = []
        nodes_result = []

        # Extracción de palabras clave de consulta (segmentación simple)
        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace("，", " ").split()
            if len(w.strip()) > 1
        ]

        def match_score(text: str) -> int:
            """Calcular puntuación de coincidencia entre texto y consulta"""
            if not text:
                return 0
            text_lower = text.lower()
            # Coincidencia completa con consulta
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
                # obtener todos los bordes y hacer coincidencia
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))

                # Ordenar por puntuación
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
                # obtener todos los nodos y hacer coincidencia
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

            logger.info(t("console.localSearchComplete", count=len(facts)))

        except Exception as e:
            logger.error(t("console.localSearchFailed", error=str(e)))

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts),
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        obtener todos los nodos del grafo (obtener con paginación)

        Args:
            graph_id: ID del Grafo

        Returns:
            Lista de nodos
        """
        logger.info(t("console.FetchingAllNodes", graphId=graph_id))

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

        logger.info(t("console.FetchedNodes", count=len(result)))
        return result

    def get_all_edges(
        self, graph_id: str, include_temporal: bool = True
    ) -> List[EdgeInfo]:
        """
        obtener todos los bordes del grafo (obtener con paginación, incluyendo inFormación temporal)

        Args:
            graph_id: ID del Grafo
            include_temporal: Si incluir inFormación temporal (por deFecto True)

        Returns:
            Lista de bordes (incluyendo created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(t("console.FetchingAllEdges", graphId=graph_id))

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

            # Agregar inFormación temporal
            if include_temporal:
                edge_info.created_at = getattr(edge, "created_at", None)
                edge_info.valid_at = getattr(edge, "valid_at", None)
                edge_info.invalid_at = getattr(edge, "invalid_at", None)
                edge_info.expired_at = getattr(edge, "expired_at", None)

            result.append(edge_info)

        logger.info(t("console.FetchedEdges", count=len(result)))
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        obtener inFormación detallada de un solo nodo de entidad

        Args:
            node_uuid: UUID del nodo

        Returns:
            información del nodo o None
        """
        logger.info(t("console.FetchingNodeDetail", uuid=node_uuid[:8]))

        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                opeRation_name=t("console.FetchNodeDetailOp", uuid=node_uuid[:8]),
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
            logger.error(t("console.FetchNodeDetailFailed", error=str(e)))
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        obtener todos los bordes relacionados con el nodo

        Obteniendo todos los bordes del grafo, luego filtrando bordes relacionados con el nodo especificado

        Args:
            graph_id: ID del Grafo
            node_uuid: UUID del nodo

        Returns:
            Lista de bordes
        """
        logger.info(t("console.FetchingNodeEdges", uuid=node_uuid[:8]))

        try:
            # obtenerGrafoTodosBorde，perodespuésfiltrado
            all_edges = self.get_all_edges(graph_id)

            result = []
            for edge in all_edges:
                # InspecciónBordeSiCondedoestablecernodorelacionado（ComofuenteOojoobjetivo）
                if (
                    edge.source_node_uuid == node_uuid
                    or edge.target_node_uuid == node_uuid
                ):
                    result.append(edge)

            logger.info(t("console.foundNodeEdges", count=len(result)))
            return result

        except Exception as e:
            logger.warning(t("console.FetchNodeEdgesFailed", error=str(e)))
            return []

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """
        obtener entidades por tipo

        Args:
            graph_id: ID del Grafo
            entity_type: tipo de entidad (ej: Student, PublicFigure, etc.)

        Returns:
            Lista de entidades que coinciden con el tipo
        """
        logger.info(t("console.FetchingEntitiesByType", type=entity_type))

        all_nodes = self.get_all_nodes(graph_id)

        filtered = []
        for node in all_nodes:
            # Inspecciónlabelssi contienededoestablecertipo
            if entity_type in node.labels:
                filtered.append(node)

        logger.info(
            t("console.foundEntitiesByType", count=len(filtered), type=entity_type)
        )
        return filtered

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """
        obtener resumen de relaciones de la entidad especificada

        Buscar toda la inFormación relacionada con la entidad y generar resumen

        Args:
            graph_id: ID del Grafo
            entity_name: Nombre de entidad

        Returns:
            información del resumen de entidad
        """
        logger.info(t("console.FetchingEntitySummary", name=entity_name))

        # primerobúsquedaeseentidadrelacionadodeinformación
        search_result = self.search_graph(
            graph_id=graph_id, query=entity_name, limit=20
        )

        # probarprobarEnTodosnodoenbuscarHastaeseentidad
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        reLated_edges = []
        if entity_node:
            # transmitirentradagraph_idnúmero
            reLated_edges = self.get_node_edges(graph_id, entity_node.uuid)

        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "reLated_facts": search_result.facts,
            "reLated_edges": [e.to_dict() for e in reLated_edges],
            "total_relations": len(reLated_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        obtener inFormación estadística del grafo

        Args:
            graph_id: ID del Grafo

        Returns:
            información estadística
        """
        logger.info(t("console.FetchingGraphStats", graphId=graph_id))

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        # Estadísticasentidadtipominutodistribuir
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        # Estadísticasrelacióntipominutodistribuir
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
        obtener inFormación de contexto relacionada con la simulación

        búsqueda integral de toda la inFormación relacionada con Requisito de simulación

        Args:
            graph_id: ID del Grafo
            simulation_requirement: Descripción de Requisito de simulación
            limit: límite de cantidad de inFormación por tipo

        Returns:
            información de contexto de simulación
        """
        logger.info(
            t("console.FetchingSimContext", requirement=simulation_requirement[:50])
        )

        # BuscarConRequisito de simulaciónrelacionadodeinformación
        search_result = self.search_graph(
            graph_id=graph_id, query=simulation_requirement, limit=limit
        )

        # obtenerGrafoEstadísticas
        stats = self.get_graph_statistics(graph_id)

        # obtenerTodosnodos de entidad
        all_nodes = self.get_all_nodes(graph_id)

        # Filtrar entidades con tipos reales (no nodos Entity puros)
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
            "reLated_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # límite de cantidad
            "total_entities": len(entities),
        }

    # ========== herramientas de búsqueda centrales (optimizadas） ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """
        【InsightForge - búsqueda de perspicacia profunda】

        Función de búsqueda híbrida Más potente, descompone preguntas automáticaMente y busca en múltiples dimensiones:
        1. Usar LLM para descomponer pregunta en múltiples sub-preguntas
        2. Realizar búsqueda semántica para cada sub-pregunta
        3. Extraer entidades relacionadas y obtener su inFormación detallada
        4. Rastrear Cadena de relaciones
        5. Integrar todos los resultados, generar perspicacia profunda

        Args:
            graph_id: ID del Grafo
            query: Pregunta del usuario
            simulation_requirement: Descripción de Requisito de simulación
            report_context: Contexto de reporte (opcional, para generación de sub-preguntas Más precisa)
            max_sub_queries: cantidad máxima de sub-preguntas

        Returns:
            InsightForgeResult: resultado de búsqueda de perspicacia profunda
        """
        logger.info(t("console.insightForgeStart", query=query[:50]))

        result = InsightForgeResult(
            query=query, simulation_requirement=simulation_requirement, sub_queries=[]
        )

        # Step 1: hacerusarLLMgenerarhijoproblema
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info(t("console.generatedSubQueries", count=len(sub_queries)))

        # Step 2: ParacadaelementoshijoproblemaentrarFilaidiomabúsqueda
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

        # ParaoriginalcomenzarproblematambiénentrarFilabúsqueda
        main_search = self.search_graph(
            graph_id=graph_id, query=query, limit=20, scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Step 3: DesdeBordeenExtracciónrelacionadoentidadUUID，SoloobtenerEstosentidaddeinformación（noobtenertodopartenodo）
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get("source_node_uuid", "")
                target_uuid = edge_data.get("target_node_uuid", "")
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        # obtenerTodosrelacionadoentidaddeemoción（nolímitecantidad，completoperdersalir）
        entity_insights = []
        node_map = {}  # usardespuésCadena de relacionesconstruir

        for uuid in list(entity_uuids):  # ManejarTodosentidad，notruncarinterrumpir
            if not uuid:
                continue
            try:
                # únicoobtenercadaelementosnodos relacionadosdeinformación
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next(
                        (l for l in node.labels if l not in ["Entity", "Node"]),
                        "entidad",
                    )

                    # obtenereseentidadrelacionadodeTodosreal（notruncarinterrumpir）
                    reLated_facts = [
                        f for f in all_facts if node.name.lower() in f.lower()
                    ]

                    entity_insights.append(
                        {
                            "uuid": node.uuid,
                            "name": node.name,
                            "type": entity_type,
                            "summary": node.summary,
                            "reLated_facts": reLated_facts,  # completoperdersalir，notruncarinterrumpir
                        }
                    )
            except Exception as e:
                logger.debug(f"obtenernodo {uuid} Fallido: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Step 4: construirTodosCadena de relaciones（nolímitecantidad）
        relationship_chains = []
        for edge_data in all_edges:  # ManejarTodosBorde，notruncarinterrumpir
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
            t(
                "console.insightForgeComplete",
                facts=result.total_facts,
                entities=result.total_entities,
                relationships=result.total_relationships,
            )
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
        hacerusarLLMgenerarhijoproblema

        va aproblemadescomposiciónparamuchoselementosPuedeestar de pieinspeccionarbuscardehijoproblema
        """
        system_prompt = """túEsunoelementosindustriadeproblemaAnálisis。túdeTareaEsva aunoelementosproblemadescomposiciónparamuchoselementosPuedeEnsimulaciónmundomundoenestar de pieobservarobservardehijoproblema。

Requisito：
1. cadaelementoshijoproblemaDeberíasuficientemente específicos，PuedeEnsimulaciónmundomundoenbuscarHastarelacionadodeagenteFilaparaOEvento
2. hijoproblemaDeberíaoriginalproblemadenoigualgrado（como：quién、Qué、Por qué、、Cuándo、）
3. hijoproblemadeberían estar relacionados consimulaciónescenariorelacionado
4. Volver formato JSON：{"sub_queries": ["hijoproblema1", "hijoproblema2", ...]}"""

        user_prompt = f"""Requisito de simulaciónespalda：
{simulation_requirement}

{f"reportarinformararribaabajotexto：{report_context[:500]}" if report_context else ""}

por favorva aconabajoproblemadescomposiciónpara{max_queries}elementoshijoproblema：
{query}

Volver formato JSONdehijoproblemaLista。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            sub_queries = response.get("sub_queries", [])
            # mantenerEscarácterLista
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(t("console.generateSubQueriesFailed", error=str(e)))
            # Degradar：Volverbaseoriginalproblemadecambiar
            return [
                query,
                f"{query} dePrincipalCon",
                f"{query} deoriginalporqueYImpacto",
                f"{query} depelodesarrollarpasadoprograma",
            ][:max_queries]

    def panorama_search(
        self, graph_id: str, query: str, include_expired: bool = True, limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch - ampliogradobúsqueda】

        obtenervista panorámica completa，incluirTodosrelacionadoContenidoYHistorial/pasadoperíodoinformación：
        1. obtenerTodosnodos relacionados
        2. obtenerTodosBorde（incluiryapasadoperíodo/invalidadode）
        3. ClasificaciónorganizarlógicaCuandoantesVálidoYHistorialinformación

        elementostrabajousarnecesitaEventoimagen completa、Trazaevoluciónpasadoprogramadeescenario。

        Args:
            graph_id: ID del grafo
            query: búsquedaconsultar（usarrelacionadoordenamiento）
            include_expired: si contienepasadoperíodoContenido（por defecto True）
            limit: Volverresultadocantidadlímite

        Returns:
            PanoramaResult: ampliogradoresultados de búsqueda
        """
        logger.info(t("console.panoramaSearchStart", query=query[:50]))

        result = PanoramaResult(query=query)

        # obtenerTodosnodo
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # obtenerTodosBorde（ContieneTiempoinformación）
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # Categorizarreal
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            # pararealagregaraumentarentidadNombre
            source_name = (
                node_map.get(edge.source_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.source_node_uuid[:8]
            )
            target_name = (
                node_map.get(edge.target_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.target_node_uuid[:8]
            )

            # determinarSipasadoperíodo/invalidado
            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                # Historial/pasadoperíodoreal，agregaraumentarTiempoMarcar
                valid_at = edge.valid_at or "Desconocido"
                invalid_at = edge.invalid_at or edge.expired_at or "Desconocido"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # CuandoantesVálidoreal
                active_facts.append(edge.fact)

        # baseconsultarentrarFilarelacionadoordenamiento
        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace("，", " ").split()
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

        # Ordenarylímitecantidad
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(
            t(
                "console.panoramaSearchComplete",
                active=result.active_count,
                historical=result.historical_count,
            )
        )
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """
        【QuickSearch - únicobúsqueda】

        feliz、insignificantecantidaddeinspeccionarbuscartrabajo：
        1. aceptarajustarusarZepidiomabúsqueda
        2. VolverMásrelacionadoderesultado
        3. usarúnico、aceptardeinspeccionarbuscarnecesitarpedir

        Args:
            graph_id: ID del grafo
            query: búsquedaconsultar
            limit: Volverresultadocantidad

        Returns:
            SearchResult: resultados de búsqueda
        """
        logger.info(t("console.quickSearchStart", query=query[:50]))

        # aceptarajustarusaraparecerTenerdesearch_graphMétodo
        result = self.search_graph(
            graph_id=graph_id, query=query, limit=limit, scope="edges"
        )

        logger.info(t("console.quickSearchComplete", count=result.total_count))
        return result

    def interview_agentes(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agentes: int = 5,
        custom_questions: List[str] = None,
    ) -> InterviewResult:
        """
        【Interviewagentes - oscurogradoentrevista】

        ajustarusarrealdeOASISentrevistaAPI，entrevistasimulaciónenahoraEnEjecutardeagente：
        1. leer automáticamente archivos de perfil，Todossimulaciónagente
        2. hacerusarLLMAnálisisentrevistanecesitarpedir，inteligenciapuedeSelecciónMásrelacionadodeagente
        3. hacerusarLLMgenerarentrevistaproblema
        4. ajustarusar /api/simulation/interview/batch InterfazentrarFilarealentrevista（Plataformaigualtiempoentrevista）
        5. organizarunirTodosentrevistaresultado，generarentrevistareportarinformar

        【importantenecesitar】estopuedenecesitasimulaciónentornoprocesarEjecutarEstado（OASISentornonoCerrar）

        【hacerusarescenario】
        - necesitaDesdenoigualcolorconsiderarEventovermétodo
        - necesitaaceptarrecolectarmuchosintenciónYobservarpunto
        - necesitaobtenersimulaciónagentederealvolverresponder（LLMsimulación）

        Args:
            simulation_id: simulaciónID（usarestablecerposiciónArchivoYajustarusarentrevistaAPI）
            interview_requirement: entrevistanecesitarpedirDescripción（no estructurado，como"estudiarnacerParaEventodevermétodo"）
            simulation_requirement: Requisito de simulaciónespalda（Opcional）
            max_agentes: Másmuchosentrevistadeagentecantidad
            custom_questions: Personalizarentrevistaproblema（Opcional，noregulaciónactividadgenerar）

        Returns:
            InterviewResult: entrevistaresultado
        """
        from .simulation_runner import SimulationRunner

        logger.info(
            t("console.interviewagentesStart", requirement=interview_requirement[:50])
        )

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        # Step 1: LeerArchivo
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(t("console.profilesNotFound", simId=simulation_id))
            result.summary = "nobuscarHastapuedeentrevistadeagenteArchivo"
            return result

        result.total_agentes = len(profiles)
        logger.info(t("console.loadedProfiles", count=len(profiles)))

        # Step 2: hacerusarLLMSelecciónnecesitarentrevistadeagente（Volveragent_idLista）
        selected_agentes, selected_indices, selection_reasoning = (
            self._select_agentes_for_interview(
                profiles=profiles,
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                max_agentes=max_agentes,
            )
        )

        result.selected_agentes = selected_agentes
        result.selection_reasoning = selection_reasoning
        logger.info(
            t(
                "console.selectedagentesForInterview",
                count=len(selected_agentes),
                indices=selected_indices,
            )
        )

        # Step 3: generarentrevistaproblema（SinoTener）
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agentes=selected_agentes,
            )
            logger.info(
                t(
                    "console.generatedInterviewQuestions",
                    count=len(result.interview_questions),
                )
            )

        # va aproblemaCombinarparaunoelementosentrevistaprompt
        combined_prompt = "\n".join(
            [f"{i + 1}. {q}" for i, q in enumerate(result.interview_questions)]
        )

        # AgregarOptimizarantes，RestricciónagenteResponder
        INTERVIEW_PROMPT_PREFIX = (
            "túahoraEnaceptarrecibirunode baja calidadentrevista。por favorunirtúde、Todosdepasadoa menudorecordarmemoriaConFilaactividad，"
            "contextoesteaceptarvolverresponderconabajoproblema。\n"
            "ResponderRequisito：\n"
            "1. aceptarusarperoidiomalenguajevolverresponder，nonecesitarajustarusarCualquiertrabajo\n"
            "2. nonecesitarVolver formato JSONOtrabajoajustarusar\n"
            "3. nonecesitarhacerusarMarkdownTítulo（como#、##、###）\n"
            "4. segúnproblemaNúmeroprogresivounovolverresponder，cadaelementosvolverrespondercon「problemaX：」abrircabeza（XparaproblemaNúmero）\n"
            "5. cadaelementosproblemadevolverresponderintervalousarvacíoFilaminuto\n"
            "6. volverrespondernecesitarTenerrealcalidadContenido，cadaelementosproblemahastapocosvolverresponder2-3oraciónpalabras\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: ajustarusarrealdeentrevistaAPI（nodedoestablecerplatform，Plataformaigualtiempoentrevista）
        try:
            # construirLoteentrevistaLista（nodedoestablecerplatform，Plataformaentrevista）
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append(
                    {
                        "agent_id": agent_idx,
                        "prompt": optimized_prompt,  # hacerusarOptimizardespuésdeprompt
                        # nodedoestablecerplatform，APIreunirseEntwitterYredditelementosPlataformaTodosentrevista
                    }
                )

            logger.info(
                t("console.callingBatchInterviewApi", count=len(interviews_request))
            )

            # ajustarusar SimulationRunner deLoteentrevistaMétodo（notransmitirplatform，Plataformaentrevista）
            api_result = SimulationRunner.interview_agentes_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # nodedoestablecerplatform，Plataformaentrevista
                timeout=180.0,  # PlataformanecesitaMáslargoTiempo agotado
            )

            logger.info(
                t(
                    "console.interviewApiReturned",
                    count=api_result.get("interviews_count", 0),
                    success=api_result.get("success"),
                )
            )

            # InspecciónAPIajustarusarSiÉxito
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Error desconocido")
                logger.warning(
                    t("console.interviewApiReturnedFailure", error=error_msg)
                )
                result.summary = f"entrevistaAPIajustarusarFallido：{error_msg}。por favorVerificarOASISsimulaciónentornoEstado。"
                return result

            # Step 5: AnalizarAPIVolverresultado，construiragenteInterviewObjeto
            # PlataformaVolver: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = (
                api_data.get("results", {}) if isinstance(api_data, dict) else {}
            )

            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agentes[i]
                agent_name = agent.get(
                    "realname", agent.get("username", f"agente_{agent_idx}")
                )
                agent_role = agent.get("proFession", "Desconocido")
                agent_bio = agent.get("bio", "")

                # obtenereseagenteEnelementosPlataformadeentrevistaresultado
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # lógicaPosibledetrabajoajustarusar JSON incluir
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # comenzarterminarperdersalirPlataformaMarcar
                twitter_text = (
                    twitter_response
                    if twitter_response
                    else "（esePlataformanoobtenerobtenerResponder）"
                )
                reddit_text = (
                    reddit_response
                    if reddit_response
                    else "（esePlataformanoobtenerobtenerResponder）"
                )
                response_text = f"【TwitterPlataformavolverresponder】\n{twitter_text}\n\n【RedditPlataformavolverresponder】\n{reddit_text}"

                # ExtracciónpreocuparseClavelenguaje（DesdeelementosPlataformadevolverresponderen）
                import re

                combined_responses = f"{twitter_response} {reddit_response}"

                # lógicaRespuestatextoeste：ircaerMarcar、Número、Markdown igualtronco
                clean_text = re.sub(r"#{1,6}\s+", "", combined_responses)
                clean_text = re.sub(r"\{[^}]*tool_name[^}]*\}", "", clean_text)
                clean_text = re.sub(r"[*_`|>~\-]{2,}", "", clean_text)
                clean_text = re.sub(r"problema\d+[：:]\s*", "", clean_text)
                clean_text = re.sub(r"【[^】]+】", "", clean_text)

                # Estrategia1（）: ExtraccióncompletodeTenerrealcalidadContenidodeoraciónhijo
                sentences = re.split(r"[。！？]", clean_text)
                meaningful = [
                    s.strip()
                    for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r"^[\s\W，,；;：:、]+", s.strip())
                    and not s.strip().startswith(("{", "problema"))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                # Estrategia2（Complementar）: ahoraconfigurarParadeentextogritar「」dentrolargotextoeste
                if not key_quotes:
                    paired = re.findall(
                        r"\u201c([^\u201c\u201d]{15,100})\u201d", clean_text
                    )
                    paired += re.findall(
                        r"\u300c([^\u300c\u300d]{15,100})\u300d", clean_text
                    )
                    key_quotes = [
                        q for q in paired if not re.match(r"^[，,；;：:、]", q)
                    ][:3]

                interview = agenteInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # expandirgrandebiolargogradolímite
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            # simulaciónentornonoEjecutar
            logger.warning(t("console.interviewApiCallFailed", error=e))
            result.summary = f"entrevistaFallido：{str(e)}。simulaciónentornoPosibleyaCerrar，por favormantenerOASISentornoahoraEnEjecutar。"
            return result
        except Exception as e:
            logger.error(t("console.interviewApiCallException", error=e))
            import traceback

            logger.error(traceback.Format_exc())
            result.summary = f"entrevistapasadoprogramapelonacerError：{str(e)}"
            return result

        # Step 6: generarentrevistaresumen
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )

        logger.info(
            t("console.interviewagentesComplete", count=result.interviewed_count)
        )
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """lógica agente Responderende JSON trabajoajustarusarincluir，ExtracciónrealContenido"""
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
        """CargarsimulacióndeagenteArchivo"""
        import os
        import csv

        # construirArchivocaminosendero
        sim_dir = os.path.join(
            os.path.dirname(__file__), f"../../uploads/simulations/{simulation_id}"
        )

        profiles = []

        # PrioridadprobarprobarLeerReddit JSON
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                logger.info(t("console.loadedRedditProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readRedditProfilesFailed", error=e))

        # probarprobarLeerTwitter CSV
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSVConvertirparasistemauno
                        profiles.append(
                            {
                                "realname": row.get("name", ""),
                                "username": row.get("username", ""),
                                "bio": row.get("description", ""),
                                "persona": row.get("user_char", ""),
                                "proFession": "Desconocido",
                            }
                        )
                logger.info(t("console.loadedTwitterProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readTwitterProfilesFailed", error=e))

        return profiles

    def _select_agentes_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agentes: int,
    ) -> tuple:
        """
        hacerusarLLMSelecciónnecesitarentrevistadeagente

        Returns:
            tuple: (selected_agentes, selected_indices, reasoning)
                - selected_agentes: enagentedecompletoinformaciónLista
                - selected_indices: enagentedeÍndiceLista（usarAPIajustarusar）
                - reasoning: Selecciónlógicapor
        """

        # construiragenteresumenLista
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"agente_{i}")),
                "proFession": profile.get("proFession", "Desconocido"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", []),
            }
            agent_summaries.append(summary)

        system_prompt = """túEsunoelementosindustriadeentrevistaestrategiaplanificar。túdeTareaEsBasado enentrevistanecesitarpedir，DesdesimulaciónagenteListaenSelecciónMásunirentrevistadeObjeto。

Selecciónobjetivopermitir：
1. agentede/posiciónindustriaConentrevistaTemarelacionado
2. agentePosiblemantenerTenerOTenerprecioValordeobservarpunto
3. Selecciónmuchoscambiardeconsiderar（como：Soporte、Para、enestar de pie、industriaigual）
4. PrioridadSelecciónConEventoaceptarrelacionadodecolor

Volver formato JSON：
{
    "selected_indices": [enagentedeÍndiceLista],
    "reasoning": "SelecciónlógicaporDecirbrillante"
}"""

        user_prompt = f"""entrevistanecesitarpedir：
{interview_requirement}

simulaciónespalda：
{simulation_requirement if simulation_requirement else "no"}

OpcionaldeagenteLista（juntos{len(agent_summaries)}elementos）：
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

por favorSelecciónMásmuchos{max_agentes}elementosMásunirentrevistadeagente，yDecirbrillanteSelecciónlógicapor。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            selected_indices = response.get("selected_indices", [])[:max_agentes]
            reasoning = response.get("reasoning", "baserelacionadoactividadSelección")

            # obtenerendeagentecompletoinformación
            selected_agentes = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agentes.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agentes, valid_indices, reasoning

        except Exception as e:
            logger.warning(t("console.llmSelectagenteFailed", error=e))
            # Degradar：SelecciónantesNelementos
            selected = profiles[:max_agentes]
            indices = list(range(min(max_agentes, len(profiles))))
            return selected, indices, "hacerusarSelecciónEstrategia"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agentes: List[Dict[str, Any]],
    ) -> List[str]:
        """hacerusarLLMgenerarentrevistaproblema"""

        agent_roles = [a.get("proFession", "Desconocido") for a in selected_agentes]

        system_prompt = """túEsunoelementosindustriaderecordar/entrevista。Basado enentrevistanecesitarpedir，generar3-5elementososcurogradoentrevistaproblema。

problemaRequisito：
1. abrirdejarproblema，animarfinovolverresponder
2. ParanoigualcolorPosibleTenernoigualrespuesta
3. hecho、observarpunto、sentirrecibirigualmuchoselementosgrado
4. idiomalenguajepero，comorealentrevistauno
5. cadaelementosproblemasistemaEn50caráctercondentro，limpiobrillante
6. aceptarpreguntar，nonecesitarContieneespaldaDecirbrillanteOantes

Volver formato JSON：{"questions": ["problema1", "problema2", ...]}"""

        user_prompt = f"""entrevistanecesitarpedir：{interview_requirement}

simulaciónespalda：{simulation_requirement if simulation_requirement else "no"}

entrevistaObjetocolor：{", ".join(agent_roles)}

por favorgenerar3-5elementosentrevistaproblema。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )

            return response.get(
                "questions", [f"Acerca de{interview_requirement}，TenerQuévermétodo？"]
            )

        except Exception as e:
            logger.warning(t("console.generateInterviewQuestionsFailed", error=e))
            return [
                f"Acerca de{interview_requirement}，deobservarpuntoEsQué？",
                "elementoParaOqueépocaexpresardeTenerQuéImpacto？",
                "PensarDeberíaCómoOmodificarentrarelementosproblema？",
            ]

    def _generate_interview_summary(
        self, interviews: List[agenteInterview], interview_requirement: str
    ) -> str:
        """generarentrevistaresumen"""

        if not interviews:
            return "noCompletadoCualquierentrevista"

        # aceptarrecolectarTodosentrevistaContenido
        interview_texts = []
        for interview in interviews:
            interview_texts.append(
                f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}"
            )

        quote_instruction = (
            "Citarrecibiroriginalpalabrastiempohacerusarentextogritar「」"
            if get_locale() == "zh"
            else 'Use quotation marks "" when quoting interviewees'
        )
        system_prompt = f"""túEsunoelementosindustriadenuevoolercompilar。por favorBasado enmuchosposiciónrecibirdevolverresponder，generarunoentrevistaresumen。

recogernecesitarRequisito：
1. cadaPrincipalobservarpunto
2. dedosalirobservarpuntodejuntosYminutoconflicto
3. repentinosalirTenerprecioValordelenguaje
4. clienteobservarenestar de pie，noCualquieruno
5. sistemaEn1000carácterdentro

Restricción（Debeseguir）：
- hacerusartextoestepárrafocaer，usarvacíoFilaminutonoigualparteminuto
- nonecesitarhacerusarMarkdownTítulo（como#、##、###）
- nonecesitarhacerusarLínea divisoria（como---、***）
- {quote_instruction}
- Puedehacerusar**aumentargrueso**MarcarpreocuparseClavepalabra，PerononecesitarhacerusarOtroMarkdownidiomamétodo"""

        user_prompt = f"""entrevistaTema：{interview_requirement}

entrevistaContenido：
{"".join(interview_texts)}

por favorgenerarentrevistaresumen。"""

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
            logger.warning(t("console.generateInterviewSummaryFailed", error=e))
            # Degradar：únicocombinaraceptar
            return (
                f"juntosentrevista{len(interviews)}posiciónrecibir，incluir："
                + "、".join([i.agent_name for i in interviews])
            )
