"""
Servicio de herramientas de búsqueda Zep
Encapsula búsqueda en grafo, lectura de nodos, consulta de aristas, etc., para uso del Report agente

Herramientas de búsqueda principales (optimizadas):
1. InsightForge (Búsqueda de mendalamiento profundo) - Búsqueda híbrida más potente, genera sub-preguntas automáticamente y busca en múltiples dimensiones
2. PanoramaSearch (Búsqueda panorámica) - Obtiene visión completa, incluyendo contenido expirado
3. QuickSearch (Búsqueda rápida) - Búsqueda rápida
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

# Intentar importar el sistema de prompts i18n (fallback a strings vacíos si no existe)
try:
    from ..prompts import load_prompt as _load_prompt

    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False
    _load_prompt = None

logger = get_logger("mirofish.zep_tools")


@dataclass
class SearchResult:
    """Resultado de búsqueda"""

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
        """Convertir a formato de texto, para comprensión del LLM"""
        text_parts = [
            f"Consulta de búsqueda: {self.query}",
            f"Se encontraron {self.total_count} elementos relacionados",
        ]

        if self.facts:
            text_parts.append("\n### 相关hecho:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Información del nodo"""

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
    """Información de arista"""

    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # 时间información
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
            f"Relación: {source} --[{self.name}]--> {target}\nHecho: {self.fact}"
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
        """¿Ha expirado?"""
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        """¿Ha失效?"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Resultado de búsqueda de profundidade (InsightForge)
    Contiene resultados de múltiples sub-preguntas y análisis integrado
    """

    query: str
    simulation_requirement: str
    sub_queries: List[str]

    # Resultados de búsqueda por dimensión
    semantic_facts: List[str] = field(
        default_factory=list
    )  # Resultados de búsqueda semántica
    entity_insights: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Perspectivas de entidad
    relationship_chains: List[str] = field(default_factory=list)  # Cadenas de relación

    # Estadísticas
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
        """Convertir a formato de texto detallado, para comprensión del LLM"""
        text_parts = [
            f"## Análisis de predicción futura",
            f"Problema de análisis: {self.query}",
            f"Escenario de predicción: {self.simulation_requirement}",
            f"\n### Estadísticas de datos de predicción",
            f"- Hechos de predicción relacionados: {self.total_facts}",
            f"- Entidades involucradas: {self.total_entities}",
            f"- Cadenas de relación: {self.total_relationships}",
        ]

        # Sub-preguntas
        if self.sub_queries:
            text_parts.append(f"\n### Sub-preguntas analizadas")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")

        # Resultados de búsqueda semántica
        if self.semantic_facts:
            text_parts.append(
                f"\n### 【Hechos clave】(Por favor cite estos原文en el informe)"
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

        # Cadenas de relación
        if self.relationship_chains:
            text_parts.append(f"\n### 【Cadenas de relación】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")

        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    panorámicobúsquedaresultado (Panorama)
    包含所有相关información，包括expirado内容
    """

    query: str

    # 全部nodo
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # 全部arista（包括expirado的）
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # 当前válido的hecho
    active_facts: List[str] = field(default_factory=list)
    # 已expirado/失效的hecho（histórico记录）
    historical_facts: List[str] = field(default_factory=list)

    # 统计
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
        """转换为文本格式（completo版本，不截断）"""
        text_parts = [
            f"## panorámicobúsquedaresultado（未来全景视图）",
            f"consulta: {self.query}",
            f"\n### 统计información",
            f"- 总nodo数: {self.total_nodes}",
            f"- 总arista数: {self.total_edges}",
            f"- 当前válidohecho: {self.active_count}条",
            f"- histórico/expiradohecho: {self.historical_count}条",
        ]

        # 当前válido的hecho（completo输出，不截断）
        if self.active_facts:
            text_parts.append(f"\n### 【当前válidohecho】(Simulaciónresultado原文)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # histórico/expiradohecho（completo输出，不截断）
        if self.historical_facts:
            text_parts.append(f"\n### 【histórico/expiradohecho】(演变过程记录)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f'{i}. "{fact}"')

        # 关键entidad（completo输出，不截断）
        if self.all_nodes:
            text_parts.append(f"\n### 【涉及entidad】")
            for node in self.all_nodes:
                entity_type = next(
                    (l for l in node.labels if l not in ["Entity", "Node"]), "entidad"
                )
                text_parts.append(f"- **{node.name}** ({entity_type})")

        return "\n".join(text_parts)


@dataclass
class agenteInterview:
    """单个agente的entrevistaresultado"""

    agent_name: str
    agent_role: str  # 角色tipo（如：学生、教师、媒体等）
    agent_bio: str  # 简介
    question: str  # entrevistapregunta
    response: str  # entrevistarespuesta
    key_quotes: List[str] = field(default_factory=list)  # 关键引言

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
        # 显示completo的agent_bio，不截断
        text += f"_简介: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**关键引言:**\n"
            for quote in self.key_quotes:
                # 清理各种引号
                clean_quote = (
                    quote.replace("\u201c", "").replace("\u201d", "").replace('"', "")
                )
                clean_quote = clean_quote.replace("\u300c", "").replace("\u300d", "")
                clean_quote = clean_quote.strip()
                # 去掉开头的标点
                while clean_quote and clean_quote[0] in "，,；;：:、。！？\n\r\t ":
                    clean_quote = clean_quote[1:]
                # 过滤包含pregunta编号的垃圾内容（pregunta1-9）
                skip = False
                for d in "123456789":
                    if f"\u95ee\u9898{d}" in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # 截断过长内容（按句号截断，而非硬截断）
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
    entrevistaresultado (Interview)
    包含多个Simulaciónagente的entrevistarespuesta
    """

    interview_topic: str  # entrevista主题
    interview_questions: List[str]  # entrevistapregunta列表

    # entrevista选择的agente
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # 各agente的entrevistarespuesta
    interviews: List[agenteInterview] = field(default_factory=list)

    # 选择agente的理由
    selection_reasoning: str = ""
    # 整合后的entrevista摘要
    summary: str = ""

    # 统计
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
        """转换为详细的文本格式，供LLM理解和报告引用"""
        text_parts = [
            "## profundoentrevista报告",
            f"**entrevista主题:** {self.interview_topic}",
            f"**entrevista人数:** {self.interviewed_count} / {self.total_agents} 位Simulaciónagente",
            "\n### entrevista对象选择理由",
            self.selection_reasoning or "（自动选择）",
            "\n---",
            "\n### entrevista实录",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### entrevista #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("（无entrevista记录）\n\n---")

        text_parts.append("\n### entrevista摘要与核心观点")
        text_parts.append(self.summary or "（无摘要）")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep检索工具服务

    【核心检索工具 - 优化后】
    1. insight_forge - profundo洞察检索（最强大，自动generarsub-pregunta，多维度检索）
    2. panorama_search - panorámicobúsqueda（obtener全貌，包括expirado内容）
    3. quick_search - 简单búsqueda（快速检索）
    4. interview_agents - profundoentrevista（entrevistaSimulaciónagente，obtener多视角观点）

    【基础工具】
    - search_graph - 图谱语义búsqueda
    - get_all_nodes - obtener图谱所有nodo
    - get_all_edges - obtener图谱所有arista（含时间información）
    - get_node_detail - obtenernodo详细información
    - get_node_edges - obtenernodo相关的arista
    - get_entities_by_type - 按tipoobtenerentidad
    - get_entity_summary - obtenerentidad的relación摘要
    """

    # reintento配置
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(
        self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None
    ):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未配置")

        self.client = Zep(api_key=self.api_key)
        # LLM客户端用于InsightForgegenerarsub-pregunta
        self._llm_client = llm_client
        logger.info("ZepToolsService 初始化完成")

    @property
    def llm(self) -> LLMClient:
        """延迟初始化LLM客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """带reintento机制的API调用"""
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
                        f"Zep {operation_name} 第 {attempt + 1} 次intentofallido: {str(e)[:100]}, "
                        f"{delay:.1f}秒后reintento..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"Zep {operation_name} 在 {max_retries} 次intento后仍fallido: {str(e)}"
                    )

        raise last_exception

    def search_graph(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        """
        图谱语义búsqueda

        使用混合búsqueda（语义+BM25）在图谱中búsqueda相关información。
        如果Zep Cloud的search API不可用，则降级为本地关键词匹配。

        Args:
            graph_id: 图谱ID (Standalone Graph)
            query: búsquedaconsulta
            limit: 返回resultado数量
            scope: búsqueda范围，"edges" 或 "nodes"

        Returns:
            SearchResult: búsquedaresultado
        """
        logger.info(f"图谱búsqueda: graph_id={graph_id}, query={query[:50]}...")

        # intento使用Zep Cloud Search API
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder",
                ),
                operation_name=f"图谱búsqueda(graph={graph_id})",
            )

            facts = []
            edges = []
            nodes = []

            # 解析aristabúsquedaresultado
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

            # 解析nodobúsquedaresultado
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
                    # nodo摘要也算作hecho
                    if hasattr(node, "summary") and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(f"búsqueda完成: encontrar {len(facts)} 条相关hecho")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )

        except Exception as e:
            logger.warning(f"Zep Search APIfallido，降级为本地búsqueda: {str(e)}")
            # 降级：使用本地关键词匹配búsqueda
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self, graph_id: str, query: str, limit: int = 10, scope: str = "edges"
    ) -> SearchResult:
        """
        本地关键词匹配búsqueda（作为Zep Search API的降级方案）

        obtener所有arista/nodo，然后在本地进行关键词匹配

        Args:
            graph_id: 图谱ID
            query: búsquedaconsulta
            limit: 返回resultado数量
            scope: búsqueda范围

        Returns:
            SearchResult: búsquedaresultado
        """
        logger.info(f"使用本地búsqueda: query={query[:30]}...")

        facts = []
        edges_result = []
        nodes_result = []

        # 提取consulta关键词（简单分词）
        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(",", " ").replace("，", " ").split()
            if len(w.strip()) > 1
        ]

        def match_score(text: str) -> int:
            """计算文本与consulta的匹配分数"""
            if not text:
                return 0
            text_lower = text.lower()
            # 完全匹配consulta
            if query_lower in text_lower:
                return 100
            # 关键词匹配
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score

        try:
            if scope in ["edges", "both"]:
                # obtener所有arista并匹配
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))

                # 按分数排序
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
                # obtener所有nodo并匹配
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

            logger.info(f"本地búsqueda完成: encontrar {len(facts)} 条相关hecho")

        except Exception as e:
            logger.error(f"本地búsquedafallido: {str(e)}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts),
        )

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        obtener图谱的所有nodo（分页obtener）

        Args:
            graph_id: 图谱ID

        Returns:
            nodo列表
        """
        logger.info(f"obtener图谱 {graph_id} 的所有nodo...")

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

        logger.info(f"obtener到 {len(result)} 个nodo")
        return result

    def get_all_edges(
        self, graph_id: str, include_temporal: bool = True
    ) -> List[EdgeInfo]:
        """
        obtener图谱的所有arista（分页obtener，包含时间información）

        Args:
            graph_id: 图谱ID
            include_temporal: 是否包含时间información（默认True）

        Returns:
            arista列表（包含created_at, valid_at, invalid_at, expired_at）
        """
        logger.info(f"obtener图谱 {graph_id} 的所有arista...")

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

            # 添加时间información
            if include_temporal:
                edge_info.created_at = getattr(edge, "created_at", None)
                edge_info.valid_at = getattr(edge, "valid_at", None)
                edge_info.invalid_at = getattr(edge, "invalid_at", None)
                edge_info.expired_at = getattr(edge, "expired_at", None)

            result.append(edge_info)

        logger.info(f"obtener到 {len(result)} 条arista")
        return result

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        obtener单个nodo的详细información

        Args:
            node_uuid: nodoUUID

        Returns:
            nodoinformación或None
        """
        logger.info(f"obtenernodo详情: {node_uuid[:8]}...")

        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"obtenernodo详情(uuid={node_uuid[:8]}...)",
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
            logger.error(f"obtenernodo详情fallido: {str(e)}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        obtenernodo相关的所有arista

        通过obtener图谱所有arista，然后过滤出与指定nodo相关的arista

        Args:
            graph_id: 图谱ID
            node_uuid: nodoUUID

        Returns:
            arista列表
        """
        logger.info(f"obtenernodo {node_uuid[:8]}... 的相关arista")

        try:
            # obtener图谱所有arista，然后过滤
            all_edges = self.get_all_edges(graph_id)

            result = []
            for edge in all_edges:
                # 检查arista是否与指定nodo相关（作为源或目标）
                if (
                    edge.source_node_uuid == node_uuid
                    or edge.target_node_uuid == node_uuid
                ):
                    result.append(edge)

            logger.info(f"encontrar {len(result)} 条与nodo相关的arista")
            return result

        except Exception as e:
            logger.warning(f"obtenernodoaristafallido: {str(e)}")
            return []

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """
        按tipoobtenerentidad

        Args:
            graph_id: 图谱ID
            entity_type: entidadtipo（如 Student, PublicFigure 等）

        Returns:
            符合tipo的entidad列表
        """
        logger.info(f"obtenertipo为 {entity_type} 的entidad...")

        all_nodes = self.get_all_nodes(graph_id)

        filtered = []
        for node in all_nodes:
            # 检查labels是否包含指定tipo
            if entity_type in node.labels:
                filtered.append(node)

        logger.info(f"encontrar {len(filtered)} 个 {entity_type} tipo的entidad")
        return filtered

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """
        obtener指定entidad的relación摘要

        búsqueda与该entidad相关的所有información，并generar摘要

        Args:
            graph_id: 图谱ID
            entity_name: entidad名称

        Returns:
            entidad摘要información
        """
        logger.info(f"obtenerentidad {entity_name} 的relación摘要...")

        # 先búsqueda该entidad相关的información
        search_result = self.search_graph(
            graph_id=graph_id, query=entity_name, limit=20
        )

        # intento在所有nodo中encontrar该entidad
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break

        related_edges = []
        if entity_node:
            # 传入graph_id参数
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
        obtener图谱的统计información

        Args:
            graph_id: 图谱ID

        Returns:
            统计información
        """
        logger.info(f"obtener图谱 {graph_id} 的统计información...")

        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)

        # 统计entidadtipo分布
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1

        # 统计relacióntipo分布
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
        obtenerSimulación相关的contextoinformación

        综合búsqueda与Simulaciónrequisito相关的所有información

        Args:
            graph_id: 图谱ID
            simulation_requirement: Simulaciónrequisito描述
            limit: 每类información的数量限制

        Returns:
            Simulacióncontextoinformación
        """
        logger.info(f"obtenerSimulacióncontexto: {simulation_requirement[:50]}...")

        # búsqueda与Simulaciónrequisito相关的información
        search_result = self.search_graph(
            graph_id=graph_id, query=simulation_requirement, limit=limit
        )

        # obtener图谱统计
        stats = self.get_graph_statistics(graph_id)

        # obtener所有entidadnodo
        all_nodes = self.get_all_nodes(graph_id)

        # 筛选有实际tipo的entidad（非纯Entitynodo）
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
            "entities": entities[:limit],  # 限制数量
            "total_entities": len(entities),
        }

    # ========== 核心检索工具（优化后） ==========

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """
        【InsightForge - profundo洞察检索】

        最强大的混合检索函数，自动分解pregunta并多维度检索：
        1. 使用LLM将pregunta分解为多个sub-pregunta
        2. 对每个sub-pregunta进行语义búsqueda
        3. 提取相关entidad并obtener其详细información
        4. 追踪relación链
        5. 整合所有resultado，generarprofundo洞察

        Args:
            graph_id: 图谱ID
            query: 用户pregunta
            simulation_requirement: Simulaciónrequisito描述
            report_context: 报告contexto（可选，用于更精准的sub-preguntagenerar）
            max_sub_queries: 最大sub-pregunta数量

        Returns:
            InsightForgeResult: profundo洞察检索resultado
        """
        logger.info(f"InsightForge profundo洞察检索: {query[:50]}...")

        result = InsightForgeResult(
            query=query, simulation_requirement=simulation_requirement, sub_queries=[]
        )

        # Step 1: 使用LLMgenerarsub-pregunta
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info(f"generar {len(sub_queries)} 个sub-pregunta")

        # Step 2: 对每个sub-pregunta进行语义búsqueda
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

        # 对原始pregunta也进行búsqueda
        main_search = self.search_graph(
            graph_id=graph_id, query=query, limit=20, scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Step 3: 从arista中提取相关entidadUUID，只obtener这些entidad的información（不obtener全部nodo）
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get("source_node_uuid", "")
                target_uuid = edge_data.get("target_node_uuid", "")
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)

        # obtener所有相关entidad的详情（不限制数量，completo输出）
        entity_insights = []
        node_map = {}  # 用于后续relación链构建

        for uuid in list(entity_uuids):  # 处理所有entidad，不截断
            if not uuid:
                continue
            try:
                # 单独obtener每个相关nodo的información
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next(
                        (l for l in node.labels if l not in ["Entity", "Node"]),
                        "entidad",
                    )

                    # obtener该entidad相关的所有hecho（不截断）
                    related_facts = [
                        f for f in all_facts if node.name.lower() in f.lower()
                    ]

                    entity_insights.append(
                        {
                            "uuid": node.uuid,
                            "name": node.name,
                            "type": entity_type,
                            "summary": node.summary,
                            "related_facts": related_facts,  # completo输出，不截断
                        }
                    )
            except Exception as e:
                logger.debug(f"obtenernodo {uuid} fallido: {e}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        # Step 4: 构建所有relación链（不限制数量）
        relationship_chains = []
        for edge_data in all_edges:  # 处理所有arista，不截断
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
            f"InsightForge完成: {result.total_facts}条hecho, {result.total_entities}个entidad, {result.total_relationships}条relación"
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
        使用LLMgenerarsub-pregunta

        将复杂pregunta分解为多个可以独立检索的sub-pregunta
        """
        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            system_prompt = _load_prompt("zep", "sub_query_generation")
        else:
            # Fallback al original en chino
            system_prompt = """你是一个专业的preguntaanálisis专家。你的任务是将一个复杂pregunta分解为多个可以在Simulación世界中独立观察的sub-pregunta。

要求：
1. 每个sub-pregunta应该足够具体，可以在Simulación世界中encontrar相关的agente行为或事件
2. sub-pregunta应该覆盖原pregunta的不同维度（如：谁、什么、为什么、怎么样、何时、何地）
3. sub-pregunta应该与Simulación场景相关
4. 返回JSON格式：{"sub_queries": ["sub-pregunta1", "sub-pregunta2", ...]}"""

        user_prompt = f"""Simulaciónrequisito背景：
{simulation_requirement}

{f"报告contexto：{report_context[:500]}" if report_context else ""}

请将以下pregunta分解为{max_queries}个sub-pregunta：
{query}

返回JSON格式的sub-pregunta列表。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            sub_queries = response.get("sub_queries", [])
            # 确保是字符串列表
            return [str(sq) for sq in sub_queries[:max_queries]]

        except Exception as e:
            logger.warning(
                f"generarsub-preguntafallido: {str(e)}，使用默认sub-pregunta"
            )
            # 降级：返回基于原pregunta的变体
            return [
                query,
                f"{query} 的主要参与者",
                f"{query} 的原因和影响",
                f"{query} 的发展过程",
            ][:max_queries]

    def panorama_search(
        self, graph_id: str, query: str, include_expired: bool = True, limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch - panorámicobúsqueda】

        obtener全貌视图，包括所有相关内容和histórico/expiradoinformación：
        1. obtener所有相关nodo
        2. obtener所有arista（包括已expirado/失效的）
        3. 分类整理当前válido和históricoinformación

        这个工具适用于需要了解事件全貌、追踪演变过程的场景。

        Args:
            graph_id: 图谱ID
            query: búsquedaconsulta（用于相关性排序）
            include_expired: 是否包含expirado内容（默认True）
            limit: 返回resultado数量限制

        Returns:
            PanoramaResult: panorámicobúsquedaresultado
        """
        logger.info(f"PanoramaSearch panorámicobúsqueda: {query[:50]}...")

        result = PanoramaResult(query=query)

        # obtener所有nodo
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        # obtener所有arista（包含时间información）
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        # 分类hecho
        active_facts = []
        historical_facts = []

        for edge in all_edges:
            if not edge.fact:
                continue

            # 为hecho添加entidad名称
            source_name = (
                node_map.get(edge.source_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.source_node_uuid[:8]
            )
            target_name = (
                node_map.get(edge.target_node_uuid, NodeInfo("", "", [], "", {})).name
                or edge.target_node_uuid[:8]
            )

            # 判断是否expirado/失效
            is_historical = edge.is_expired or edge.is_invalid

            if is_historical:
                # histórico/expiradohecho，添加时间标记
                valid_at = edge.valid_at or "Desconocido"
                invalid_at = edge.invalid_at or edge.expired_at or "Desconocido"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # 当前válidohecho
                active_facts.append(edge.fact)

        # 基于consulta进行相关性排序
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

        # 排序并限制数量
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(
            f"PanoramaSearch完成: {result.active_count}条válido, {result.historical_count}条histórico"
        )
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """
        【QuickSearch - 简单búsqueda】

        快速、轻量级的检索工具：
        1. 直接调用Zep语义búsqueda
        2. 返回最相关的resultado
        3. 适用于简单、直接的检索requisito

        Args:
            graph_id: 图谱ID
            query: búsquedaconsulta
            limit: 返回resultado数量

        Returns:
            SearchResult: búsquedaresultado
        """
        logger.info(f"QuickSearch 简单búsqueda: {query[:50]}...")

        # 直接调用现有的search_graph方法
        result = self.search_graph(
            graph_id=graph_id, query=query, limit=limit, scope="edges"
        )

        logger.info(f"QuickSearch完成: {result.total_count}条resultado")
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
        【Interviewagentes - profundoentrevista】

        调用真实的OASISentrevistaAPI，entrevistaSimulación中正在运行的agente：
        1. 自动读取人设文件，了解所有Simulaciónagente
        2. 使用LLManálisisentrevistarequisito，智能选择最相关的agente
        3. 使用LLMgenerarentrevistapregunta
        4. 调用 /api/simulation/interview/batch 接口进行真实entrevista（双平台同时entrevista）
        5. 整合所有entrevistaresultado，generarentrevista报告

        【重要】此功能需要Simulación环境处于运行状态（OASIS环境未关闭）

        【使用场景】
        - 需要从不同角色视角了解事件看法
        - 需要收集多方意见和观点
        - 需要obtenerSimulaciónagente的真实respuesta（非LLMSimulación）

        Args:
            simulation_id: SimulaciónID（用于定位人设文件和调用entrevistaAPI）
            interview_requirement: entrevistarequisito描述（非结构化，如"了解学生对事件的看法"）
            simulation_requirement: Simulaciónrequisito背景（可选）
            max_agents: 最多entrevista的agente数量
            custom_questions: 自定义entrevistapregunta（可选，若不提供则自动generar）

        Returns:
            InterviewResult: entrevistaresultado
        """
        from .simulation_runner import SimulationRunner

        logger.info(
            f"Interviewagentes profundoentrevista（真实API）: {interview_requirement[:50]}..."
        )

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        # Step 1: 读取人设文件
        profiles = self._load_agent_profiles(simulation_id)

        if not profiles:
            logger.warning(f"未encontrarSimulación {simulation_id} 的人设文件")
            result.summary = "未encontrar可entrevista的agente人设文件"
            return result

        result.total_agents = len(profiles)
        logger.info(f"加载到 {len(profiles)} 个agente人设")

        # Step 2: 使用LLM选择要entrevista的agente（返回agent_id列表）
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
            f"选择了 {len(selected_agents)} 个agente进行entrevista: {selected_indices}"
        )

        # Step 3: generarentrevistapregunta（如果没有提供）
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents,
            )
            logger.info(
                f"generar了 {len(result.interview_questions)} 个entrevistapregunta"
            )

        # 将pregunta合并为一个entrevistaprompt
        combined_prompt = "\n".join(
            [f"{i + 1}. {q}" for i, q in enumerate(result.interview_questions)]
        )

        # 添加优化前缀，约束agente回复格式 (i18n)
        if _PROMPTS_AVAILABLE:
            INTERVIEW_PROMPT_PREFIX = _load_prompt("zep", "interview_prompt_prefix")
        else:
            # Fallback al original en chino
            INTERVIEW_PROMPT_PREFIX = (
                "你正在接受一次entrevista。请结合你的人设、所有的过往记忆与行动，"
                "以纯文本方式直接respuesta以下pregunta。\n"
                "回复要求：\n"
                "1. 直接用自然语言respuesta，不要调用任何工具\n"
                "2. 不要返回JSON格式或工具调用格式\n"
                "3. 不要使用Markdown标题（如#、##、###）\n"
                "4. 按pregunta编号逐一respuesta，每个respuesta以「preguntaX：」开头（X为pregunta编号）\n"
                "5. 每个pregunta的respuesta之间用空行分隔\n"
                "6. respuesta要有实质内容，每个pregunta至少respuesta2-3句话\n\n"
            )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"

        # Step 4: 调用真实的entrevistaAPI（不指定platform，默认双平台同时entrevista）
        try:
            # 构建批量entrevista列表（不指定platform，双平台entrevista）
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append(
                    {
                        "agent_id": agent_idx,
                        "prompt": optimized_prompt,  # 使用优化后的prompt
                        # 不指定platform，API会在twitter和reddit两个平台都entrevista
                    }
                )

            logger.info(
                f"调用批量entrevistaAPI（双平台）: {len(interviews_request)} 个agente"
            )

            # 调用 SimulationRunner 的批量entrevista方法（不传platform，双平台entrevista）
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # 不指定platform，双平台entrevista
                timeout=180.0,  # 双平台需要更长超时
            )

            logger.info(
                f"entrevistaAPI返回: {api_result.get('interviews_count', 0)} 个resultado, success={api_result.get('success')}"
            )

            # 检查API调用是否exitoso
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Desconocidoerror")
                logger.warning(f"entrevistaAPI返回fallido: {error_msg}")
                result.summary = f"entrevistaAPI调用fallido：{error_msg}。请检查OASISSimulación环境状态。"
                return result

            # Step 5: 解析API返回resultado，构建agenteInterview对象
            # 双平台模式返回格式: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
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

                # obtener该agente在两个平台的entrevistaresultado
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})

                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # 清理可能的工具调用 JSON 包裹
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # 始终输出双平台标记
                twitter_text = (
                    twitter_response if twitter_response else "（该平台未获得回复）"
                )
                reddit_text = (
                    reddit_response if reddit_response else "（该平台未获得回复）"
                )
                response_text = f"【Twitter平台respuesta】\n{twitter_text}\n\n【Reddit平台respuesta】\n{reddit_text}"

                # 提取关键引言（从两个平台的respuesta中）
                import re

                combined_responses = f"{twitter_response} {reddit_response}"

                # 清理响应文本：去掉标记、编号、Markdown 等干扰
                clean_text = re.sub(r"#{1,6}\s+", "", combined_responses)
                clean_text = re.sub(r"\{[^}]*tool_name[^}]*\}", "", clean_text)
                clean_text = re.sub(r"[*_`|>~\-]{2,}", "", clean_text)
                clean_text = re.sub(r"pregunta\d+[：:]\s*", "", clean_text)
                clean_text = re.sub(r"【[^】]+】", "", clean_text)

                # 策略1（主）: 提取completo的有实质内容的句子
                sentences = re.split(r"[。！？]", clean_text)
                meaningful = [
                    s.strip()
                    for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r"^[\s\W，,；;：:、]+", s.strip())
                    and not s.strip().startswith(("{", "pregunta"))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                # 策略2（补充）: 正确配对的中文引号「」内长文本
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
                    agent_bio=agent_bio[:1000],  # 扩大bio长度限制
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5],
                )
                result.interviews.append(interview)

            result.interviewed_count = len(result.interviews)

        except ValueError as e:
            # Simulación环境未运行
            logger.warning(f"entrevistaAPI调用fallido（环境未运行？）: {e}")
            result.summary = f"entrevistafallido：{str(e)}。Simulación环境可能已关闭，请确保OASIS环境正在运行。"
            return result
        except Exception as e:
            logger.error(f"entrevistaAPI调用异常: {e}")
            import traceback

            logger.error(traceback.format_exc())
            result.summary = f"entrevista过程发生error：{str(e)}"
            return result

        # Step 6: generarentrevista摘要
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement,
            )

        logger.info(
            f"Interviewagentes完成: entrevista了 {result.interviewed_count} 个agente（双平台）"
        )
        return result

    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """清理 agente 回复中的 JSON 工具调用包裹，提取实际内容"""
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
        """加载Simulación的agente人设文件"""
        import os
        import csv

        # 构建人设文件路径
        sim_dir = os.path.join(
            os.path.dirname(__file__), f"../../uploads/simulations/{simulation_id}"
        )

        profiles = []

        # 优先intento读取Reddit JSON格式
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                logger.info(f"从 reddit_profiles.json 加载了 {len(profiles)} 个人设")
                return profiles
            except Exception as e:
                logger.warning(f"读取 reddit_profiles.json fallido: {e}")

        # intento读取Twitter CSV格式
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSV格式转换为统一格式
                        profiles.append(
                            {
                                "realname": row.get("name", ""),
                                "username": row.get("username", ""),
                                "bio": row.get("description", ""),
                                "persona": row.get("user_char", ""),
                                "profession": "Desconocido",
                            }
                        )
                logger.info(f"从 twitter_profiles.csv 加载了 {len(profiles)} 个人设")
                return profiles
            except Exception as e:
                logger.warning(f"读取 twitter_profiles.csv fallido: {e}")

        return profiles

    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> tuple:
        """
        使用LLM选择要entrevista的agente

        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: 选中agente的completoinformación列表
                - selected_indices: 选中agente的索引列表（用于API调用）
                - reasoning: 选择理由
        """

        # 构建agente摘要列表
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

        system_prompt = """你是一个专业的entrevista策划专家。你的任务是根据entrevistarequisito，从Simulaciónagente列表中选择最适合entrevista的对象。

选择标准：
1. agente的身份/职业与entrevista主题相关
2. agente可能持有独特或有价值的观点
3. 选择多样化的视角（如：支持方、反对方、中立方、专业人士等）
4. 优先选择与事件直接相关的角色

返回JSON格式：
{
    "selected_indices": [选中agente的索引列表],
    "reasoning": "选择理由说明"
}"""

        user_prompt = f"""entrevistarequisito：
{interview_requirement}

Simulación背景：
{simulation_requirement if simulation_requirement else "未提供"}

可选择的agente列表（共{len(agent_summaries)}个）：
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

请选择最多{max_agents}个最适合entrevista的agente，并说明选择理由。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "基于相关性自动选择")

            # obtener选中的agentecompletoinformación
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)

            return selected_agents, valid_indices, reasoning

        except Exception as e:
            logger.warning(f"LLM选择agentefallido，使用默认选择: {e}")
            # 降级：选择前N个
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "使用默认选择策略"

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """使用LLMgenerarentrevistapregunta"""

        agent_roles = [a.get("profession", "Desconocido") for a in selected_agents]

        system_prompt = """你是一个专业的记者/entrevista者。根据entrevistarequisito，generar3-5个profundoentrevistapregunta。

pregunta要求：
1. 开放性pregunta，鼓励详细respuesta
2. 针对不同角色可能有不同答案
3. 涵盖hecho、观点、感受等多个维度
4. 语言自然，像真实entrevista一样
5. 每个pregunta控制在50字以内，简洁明了
6. 直接提问，不要包含背景说明或前缀

返回JSON格式：{"questions": ["pregunta1", "pregunta2", ...]}"""

        user_prompt = f"""entrevistarequisito：{interview_requirement}

Simulación背景：{simulation_requirement if simulation_requirement else "未提供"}

entrevista对象角色：{", ".join(agent_roles)}

请generar3-5个entrevistapregunta。"""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )

            return response.get(
                "questions", [f"关于{interview_requirement}，您有什么看法？"]
            )

        except Exception as e:
            logger.warning(f"generarentrevistapreguntafallido: {e}")
            return [
                f"关于{interview_requirement}，您的观点是什么？",
                "这件事对您或您所代表的群体有什么影响？",
                "您认为应该如何解决或改进这个pregunta？",
            ]

    def _generate_interview_summary(
        self, interviews: List[agenteInterview], interview_requirement: str
    ) -> str:
        """generarentrevista摘要"""

        if not interviews:
            return "未完成任何entrevista"

        # 收集所有entrevista内容
        interview_texts = []
        for interview in interviews:
            interview_texts.append(
                f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}"
            )

        system_prompt = """你是一个专业的新闻编辑。请根据多位受访者的respuesta，generar一份entrevista摘要。

摘要要求：
1. 提炼各方主要观点
2. 指出观点的共识和分歧
3. 突出有价值的引言
4. 客观中立，不偏袒任何一方
5. 控制在1000字内

格式约束（必须遵守）：
- 使用纯文本段落，用空行分隔不同部分
- 不要使用Markdown标题（如#、##、###）
- 不要使用分割线（如---、***）
- 引用受访者原话时使用中文引号「」
- 可以使用**加粗**标记关键词，但不要使用其他Markdown语法"""

        user_prompt = f"""entrevista主题：{interview_requirement}

entrevista内容：
{"".join(interview_texts)}

请generarentrevista摘要。"""

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
            logger.warning(f"generarentrevista摘要fallido: {e}")
            # 降级：简单拼接
            return f"共entrevista了{len(interviews)}位受访者，包括：" + "、".join(
                [i.agent_name for i in interviews]
            )
