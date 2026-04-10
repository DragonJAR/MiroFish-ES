"""
Generador inteligente de configuración de simulación
Usar LLM para generar automáticaMente parámetros de simulación detallados según Requisito de simulación, contenido de documentos e inFormación de grafo
Implementa automatización completa, Ninguno necesita configuración manual de parámetros

Adoptar Estrategia de generación por pasos para evitar Falido por generar contenido Demasiado Largo de una vez:
1. Generar Configuración de tiempo
2. Generar Configuración de eventos
3. Generar configuración de Agente en lotes
4. Generar configuración de plataForma
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger("mirofish.simulation_config")

# Configuración de tiempo de Hábitos chinos (hora de Beijing)
CHINA_TIMEZONE_CONFIG = {
    # Horas de madrugada (casi Ninguno actividad)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Horario matutino (despertando gradualMente)
    "morning_hours": [6, 7, 8],
    # Horario laboral
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Horas pico nocturnas (Más activas)
    "peak_hours": [19, 20, 21, 22],
    # Horas nocturnas (Nivel de actividad descendiendo)
    "night_hours": [23],
    # Coeficientes de Nivel de actividad
    "activity_multipliers": {
        "dead": 0.05,  # Madrugada casi Ninguna actividad
        "morning": 0.4,  # Mañana gradualMente activo
        "work": 0.7,  # Horario laboral moderado
        "peak": 1.5,  # Pico nocturno
        "night": 0.5,  # Madrugada descendiendo
    },
}


@dataclass
class AgentActivityConfig:
    """Configuración de actividad de un solo agente"""

    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Configuración de Nivel de actividad (0.0-1.0)
    activity_level: float = 0.5  # Nivel de actividad general

    # Frecuencia de publicaciones (número esperado de publicaciones por hora)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Período de actividad (Sistema de 24horas, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Velocidad de Respuesta (retraso de reacción a eventos candentes, Unidad: minutos de simulación)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Tendencia emocional (-1.0 a 1.0, negativo a positivo)
    sentiment_bias: float = 0.0

    # Postura (Actitud hacia temas específicos)
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Peso de Influencia (determina la probabilidad de que sus publicaciones sean vistas por Otros agentes)
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Configuración de simulación de tiempo (basada en Hábitos chinos)"""

    # Duración total de simulación (número de horas de simulación)
    total_simulation_hours: int = 72  # Por deFecto simular 72horas (3 días)

    # Tiempo representado por ronda (minutos de simulación) - Por deFecto 60minutos (1hora), acelerar flujo de tiempo
    minutes_per_round: int = 60

    # Rango de cantidad de Agentes activados por hora
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Horas pico (19-22h, tiempo Más activo para chinos)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Horas valle (0-5h, casi Ninguna actividad)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = (
        0.05  # Nivel de actividad extremadaMente Bajo en madrugada
    )

    # Horario matutino
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Horario laboral
    work_hours: List[int] = field(
        default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    )
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Configuración de eventos"""

    # Eventos iniciales (eventos desencadenados al inicio de simulación)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Eventos Programados (eventos desencadenados en tiempos específicos)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Palabras clave de temas candentes
    hot_topics: List[str] = field(default_factory=list)

    # Dirección de guía de Opinión pública
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Plataforma特定Configuración"""

    platform: str  # twitter or reddit

    # Recomendar算法Peso
    recency_weight: float = 0.4  # Tiempo新鲜度
    popularity_weight: float = 0.3  # 热度
    relevance_weight: float = 0.3  # Relacionado性

    # 病毒传播阈Valor（达HastaCuánto互动后Disparar扩散）
    vIral_threshold: int = 10

    # 回声室效应强度（相似观点聚集程度）
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """完整的Simulación参数Configuración"""

    # 基础Información
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Configuración de tiempo
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # AgentConfiguraciónLista
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Configuración de eventos
    event_config: EventConfig = field(default_factory=EventConfig)

    # PlataFormaConfiguración
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # LLMConfiguración
    llm_model: str = ""
    llm_base_url: str = ""

    # Generación元Datos
    generated_at: str = field(default_factory=lambda: datetime.now().isoFormat())
    geneRation_reasoning: str = ""  # LLM的推理Decir明

    def to_dict(self) -> Dict[str, Any]:
        """Convertir为Diccionario"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config)
            if self.twitter_config
            else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "geneRation_reasoning": self.geneRation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convertir为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    SimulaciónConfiguración智能Generar器

    使用LLMAnálisisRequisito de simulación、DocumentaciónContenido、GrafoEntidadInformación，
    自动GenerarMás佳的Simulación参数Configuración

    采用分步GenerarEstrategia：
    1. GenerarConfiguración de tiempoYConfiguración de eventos（轻量级）
    2. 分批GenerarAgentConfiguración（每批10-20Elementos）
    3. GenerarPlataformaConfiguración
    """

    # 上下文Más大字符数
    MAX_CONTEXT_LENGTH = 50000
    # 每批Generar的AgentCantidad
    AGENTS_PER_BATCH = 15

    # 各Paso的上下文截断长度（字符数）
    TIME_CONFIG_CONTEXT_LENGTH = 10000  # Configuración de tiempo
    EVENT_CONFIG_CONTEXT_LENGTH = 8000  # Configuración de eventos
    ENTITY_SUMMARY_LENGTH = 300  # Entidad摘要
    AGENT_SUMMARY_LENGTH = 300  # AgentConfiguración中的Entidad摘要
    ENTITIES_PER_TYPE_DISPLAY = 20  # 每ClaseEntidadMostrarCantidad

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未Configuración")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        智能Generar完整的SimulaciónConfiguración（分步Generar）

        Args:
            simulation_id: SimulaciónID
            project_id: 项目ID
            graph_id: GrafoID
            simulation_requirement: Requisito de simulaciónDescripción
            document_text: 原始DocumentaciónContenido
            entities: Filtrado后的EntidadLista
            enable_twitter: Si启用Twitter
            enable_reddit: Si启用Reddit
            progress_callback: 进度CallbackFunción(current_step, total_steps, message)

        Returns:
            SimulationParameters: 完整的Simulación参数
        """
        logger.info(
            f"Inicio智能Generando configuración de simulación: simulation_id={simulation_id}, Entidad数={len(entities)}"
        )

        # 计算总Paso数
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = (
            3 + num_batches
        )  # Configuración de tiempo + Configuración de eventos + N批Agent + PlataformaConfiguración
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. 构建基础上下文Información
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
        )

        reasoning_parts = []

        # ========== Paso1: GenerarConfiguración de tiempo ==========
        report_progress(1, t("progress.geneRatingTimeConfig"))
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(
            f"{t('progress.timeConfigLabel')}: {time_config_result.get('reasoning', t('common.success'))}"
        )

        # ========== Paso2: GenerarConfiguración de eventos ==========
        report_progress(2, t("progress.geneRatingEventConfig"))
        event_config_result = self._generate_event_config(
            context, simulation_requirement, entities
        )
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(
            f"{t('progress.eventConfigLabel')}: {event_config_result.get('reasoning', t('common.success'))}"
        )

        # ========== Paso3-N: 分批GenerarAgentConfiguración ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                t(
                    "progress.geneRatingAgentConfig",
                    start=start_idx + 1,
                    end=end_idx,
                    total=len(entities),
                ),
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement,
            )
            all_agent_configs.extend(batch_configs)

        reasoning_parts.append(
            t("progress.agentConfigResult", count=len(all_agent_configs))
        )

        # ========== 为初始帖子分配Publicar者 Agent ==========
        logger.info("为初始帖子分配合适的Publicar者 Agent...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len(
            [
                p
                for p in event_config.initial_posts
                if p.get("poster_agent_id") is not None
            ]
        )
        reasoning_parts.append(t("progress.postAssignResult", count=assigned_count))

        # ========== Más后一步: GenerarPlataformaConfiguración ==========
        report_progress(total_steps, t("progress.geneRatingPlatformConfig"))
        twitter_config = None
        reddit_config = None

        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                vIral_threshold=10,
                echo_chamber_strength=0.5,
            )

        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                vIral_threshold=15,
                echo_chamber_strength=0.6,
            )

        # 构建Más终参数
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            geneRation_reasoning=" | ".join(reasoning_parts),
        )

        logger.info(
            f"SimulaciónGeneración de configuración completada: {len(params.agent_configs)} ElementosAgentConfiguración"
        )

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
    ) -> str:
        """构建LLM上下文，截断HastaMás大长度"""

        # Entidad摘要
        entity_summary = self._summarize_entities(entities)

        # 构建上下文
        context_parts = [
            f"## Requisito de simulación\n{simulation_requirement}",
            f"\n## EntidadInformación ({len(entities)}Elementos)\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = (
            self.MAX_CONTEXT_LENGTH - current_length - 500
        )  # 留500字符余量

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(Documentación已截断)"
            context_parts.append(f"\n## 原始DocumentaciónContenido\n{doc_text}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """GenerarEntidad摘要"""
        lines = []

        # 按TipoAgrupar
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)}Elementos)")
            # 使用Configuración的MostrarCantidadY摘要长度
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (
                    (e.summary[:summary_len] + "...")
                    if len(e.summary) > summary_len
                    else e.summary
                )
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(
                    f"  ... TodavíaTener {len(type_entities) - display_count} Elementos"
                )

        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """带Reintentar的LLM调用，ContieneJSONCorrección逻辑"""
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_Format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),  # 每次Reintentar降低温度
                    # 不Configurarmax_tokens，让LLM自由发挥
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # InspecciónSi被截断
                if finish_reason == "length":
                    logger.warning(f"LLM输出被截断 (attempt {attempt + 1})")
                    content = self._fix_truncated_json(content)

                # 尝试AnalizarJSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"JSONAnalizarFallido (attempt {attempt + 1}): {str(e)[:80]}"
                    )

                    # 尝试CorrecciónJSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"LLM调用Fallido (attempt {attempt + 1}): {str(e)[:80]}")
                last_error = e
                import time

                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("LLM调用Fallido")

    def _fix_truncated_json(self, content: str) -> str:
        """Corrección被截断的JSON"""
        content = content.strip()

        # 计算未闭合的括号
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # InspecciónSiTener未闭合的字符串
        if content and content[-1] not in '",}]':
            content += '"'

        # 闭合括号
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """尝试CorrecciónConfiguraciónJSON"""
        import re

        # Corrección被截断的情况
        content = self._fix_truncated_json(content)

        # ExtracciónJSON部分
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # Eliminar字符串中的换Fila符
            def fix_string(match):
                s = match.group(0)
                s = s.replace("\n", " ").replace("\r", " ")
                s = re.sub(r"\s+", " ", s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # 尝试移除Todos控制字符
                json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                json_str = re.sub(r"\s+", " ", json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """GenerarConfiguración de tiempo"""
        # 使用Configuración的上下文截断长度
        context_truncated = context[: self.TIME_CONFIG_CONTEXT_LENGTH]

        # 计算Más大允许Valor（80%的agent数）
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""基于以下Requisito de simulación，GenerarTiempoSimulaciónConfiguración。

{context_truncated}

## Tarea
请GenerarConfiguración de tiempoJSON。

### Básico原则（仅供ReFerencia，需Basado en具体EventoY参Con群体灵活调整）：
- 请Basado enSimulación场景推断目标用户群体所En时区Y作息习惯，以下为东八区(UTC+8)的ReFerencia示例
- 凌晨0-5点CasiNinguno人Campaña（Nivel de actividad系数0.05）
- 早上6-8点逐渐活跃（Nivel de actividad系数0.4）
- 工作Tiempo9-18点中等活跃（Nivel de actividad系数0.7）
- 晚间19-22点Es高峰期（Nivel de actividad系数1.5）
- 23点后Nivel de actividad下降（Nivel de actividad系数0.5）
- 一般规律：凌晨低活跃、早间渐增、Horario laboral中等、晚间高峰
- **重要**：以下示例Valor仅供ReFerencia，你NecesitaBasado enEvento性质、参Con群体特点来调整具体时段
  - 例如：学生群体高峰PosibleEs21-23点；媒体全天活跃；官方机构SoloEn工作Tiempo
  - 例如：突发热点Posible导致深夜也TenerDiscusión，off_peak_hours 可适Cuando缩短

### VolverJSON格式（不要markdown）

示例：
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "针Para该Evento的Configuración de tiempoDecir明"
}}

CampoDecir明：
- total_simulation_hours (int): Simulación总时长，24-168horas，突发Evento短、持续话题长
- minutes_per_round (int): Duración por ronda，30-120minutos，Sugerencia60minutos
- agents_per_hour_min (int): 每horasMás少激活Agent数（取Valor范围: 1-{max_agents_allowed}）
- agents_per_hour_max (int): 每horasMás多激活Agent数（取Valor范围: 1-{max_agents_allowed}）
- peak_hours (int数组): Horas pico，Basado enEvento参Con群体调整
- off_peak_hours (int数组): Horas valle，通常深夜凌晨
- morning_hours (int数组): Horario matutino
- work_hours (int数组): Horario laboral
- reasoning (string): 简要Decir明Por qué这样Configuración"""

        system_prompt = "你Es社交媒体Simulación专家。Volver纯JSON格式，Configuración de tiempo需符合Simulación场景中目标用户群体的作息习惯。"
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Configuración de tiempoLLMGenerarFallido: {e}, 使用默认Configuración")
            return self._get_default_time_config(num_entities)

    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Obtener默认Configuración de tiempo（中国人作息）"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 每rondas1horas，加快Tiempo流速
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "使用默认中国人作息Configuración（每rondas1horas）",
        }

    def _parse_time_config(
        self, result: Dict[str, Any], num_entities: int
    ) -> TimeSimulationConfig:
        """AnalizarConfiguración de tiempo结果，并验证agents_per_hourValor不超过总agent数"""
        # Obtener原始Valor
        agents_per_hour_min = result.get(
            "agents_per_hour_min", max(1, num_entities // 15)
        )
        agents_per_hour_max = result.get(
            "agents_per_hour_max", max(5, num_entities // 5)
        )

        # Verificación并修正：确保不超过总agent数
        if agents_per_hour_min > num_entities:
            logger.warning(
                f"agents_per_hour_min ({agents_per_hour_min}) 超过总Agent数 ({num_entities})，已修正"
            )
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(
                f"agents_per_hour_max ({agents_per_hour_max}) 超过总Agent数 ({num_entities})，已修正"
            )
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # 确保 min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(
                f"agents_per_hour_min >= max，已修正为 {agents_per_hour_min}"
            )

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get(
                "minutes_per_round", 60
            ),  # Por deFecto每rondas1horas
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # 凌晨CasiNinguno人
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5,
        )

    def _generate_event_config(
        self, context: str, simulation_requirement: str, entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """GenerarConfiguración de eventos"""

        # Obtener可用的EntidadTipoLista，供 LLM ReFerencia
        entity_types_available = list(
            set(e.get_entity_type() or "Unknown" for e in entities)
        )

        # 为每种TipoColumna出代表性EntidadNombre
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join(
            [f"- {t}: {', '.join(examples)}" for t, examples in type_examples.items()]
        )

        # 使用Configuración的上下文截断长度
        context_truncated = context[: self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""基于以下Requisito de simulación，GenerarConfiguración de eventos。

Requisito de simulación: {simulation_requirement}

{context_truncated}

## 可用EntidadTipo及示例
{type_info}

## Tarea
请GenerarConfiguración de eventosJSON：
- Extracción热点话题关Clave词
- Descripción舆论发展方Hacia
- 设计初始帖子Contenido，**每Elementos帖子Debe指定 poster_type（Publicar者Tipo）**

**重要**: poster_type DebeDesde上面的"可用EntidadTipo"中Selección，这样初始帖子才能分配给合适的 Agent Publicar。
例如：官方声明应由 Official/University TipoPublicar，新闻由 MediaOutlet Publicar，学生观点由 Student Publicar。

VolverJSON格式（不要markdown）：
{{
    "hot_topics": ["关Clave词1", "关Clave词2", ...],
    "narrative_direction": "<舆论发展方HaciaDescripción>",
    "initial_posts": [
        {{"content": "帖子Contenido", "poster_type": "EntidadTipo（DebeDesde可用Tipo中Selección）"}},
        ...
    ],
    "reasoning": "<简要Decir明>"
}}"""

        system_prompt = "你Es舆论Análisis专家。Volver纯JSON格式。注意 poster_type Debe精确匹配可用EntidadTipo。"
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'poster_type' field value MUST be in English PascalCase exactly Matching the available entity types. Only 'content', 'narrative_direction', 'hot_topics' and 'reasoning' fields should use the specified language."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Configuración de eventosLLMGenerarFallido: {e}, 使用默认Configuración")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "使用默认Configuración",
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """AnalizarConfiguración de eventos结果"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", ""),
        )

    def _assign_initial_post_agents(
        self, event_config: EventConfig, agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        为初始帖子分配合适的Publicar者 Agent

        Basado en每Elementos帖子的 poster_type 匹配Más合适的 agent_id
        """
        if not event_config.initial_posts:
            return event_config

        # 按EntidadTipo建立 agent Índice
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # TipoMapeo表（Procesar LLM Posible输出的不同格式）
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "proFessor": ["proFessor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        # Registrar每种Tipo已使用的 agent Índice，避免重复使用同一Elementos agent
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            # 尝试找Hasta匹配的 agent
            matched_agent_id = None

            # 1. 直接匹配
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. 使用别名匹配
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break

            # 3. Si仍未找Hasta，使用InfluenciaMás高的 agent
            if matched_agent_id is None:
                logger.warning(
                    f"未找HastaTipo '{poster_type}' 的匹配 Agent，使用InfluenciaMás高的 Agent"
                )
                if agent_configs:
                    # 按InfluenciaOrdenamiento，SelecciónInfluenciaMás高的
                    sorted_agents = sorted(
                        agent_configs, key=lambda a: a.influence_weight, reverse=True
                    )
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0

            updated_posts.append(
                {
                    "content": content,
                    "poster_type": post.get("poster_type", "Unknown"),
                    "poster_agent_id": matched_agent_id,
                }
            )

            logger.info(
                f"初始帖子分配: poster_type='{poster_type}' -> agent_id={matched_agent_id}"
            )

        event_config.initial_posts = updated_posts
        return event_config

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str,
    ) -> List[AgentActivityConfig]:
        """分批GenerarAgentConfiguración"""

        # 构建EntidadInformación（使用Configuración的摘要长度）
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append(
                {
                    "agent_id": start_idx + i,
                    "entity_name": e.name,
                    "entity_type": e.get_entity_type() or "Unknown",
                    "summary": e.summary[:summary_len] if e.summary else "",
                }
            )

        prompt = f"""基于以下Información，为每ElementosEntidadGenerar社交媒体CampañaConfiguración。

Requisito de simulación: {simulation_requirement}

## EntidadLista
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Tarea
为每ElementosEntidadGenerarCampañaConfiguración，注意：
- **Tiempo符合目标用户群体作息**：以下为ReFerencia（东八区），请Basado enSimulación场景调整
- **官方机构**（University/GovernmentAgency）：Nivel de actividad低(0.1-0.3)，工作Tiempo(9-17)Campaña，Respuesta慢(60-240minutos)，Influencia高(2.5-3.0)
- **媒体**（MediaOutlet）：Nivel de actividad中(0.4-0.6)，全天Campaña(8-23)，Respuesta快(5-30minutos)，Influencia高(2.0-2.5)
- **Elementos人**（Student/Person/Alumni）：Nivel de actividad高(0.6-0.9)，Principal晚间Campaña(18-23)，Respuesta快(1-15minutos)，Influencia低(0.8-1.2)
- **公众人物/专家**：Nivel de actividad中(0.4-0.6)，Influencia中高(1.5-2.0)

VolverJSON格式（不要markdown）：
{{
    "agent_configs": [
        {{
            "agent_id": <DebeCon输入一致>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <发帖频率>,
            "comments_per_hour": <评论频率>,
            "active_hours": [<活跃horasLista，考虑中国人作息>],
            "response_delay_min": <Más小Retraso de Respuestaminutos>,
            "response_delay_max": <Más大Retraso de Respuestaminutos>,
            "sentiment_bias": <-1.0Hasta1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <InfluenciaPeso>
        }},
        ...
    ]
}}"""

        system_prompt = "你Es社交媒体Fila为Análisis专家。Volver纯JSON，Configuración需符合Simulación场景中目标用户群体的作息习惯。"
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'stance' field value MUST be one of the English strings: 'supportive', 'opposing', 'neutral', 'observer'. All JSON field names and numeric values must remain unchanged. Only natural language text fields should use the specified language."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {
                cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])
            }
        except Exception as e:
            logger.warning(f"AgentConfiguración批次LLMGenerarFallido: {e}, 使用ReglaGenerar")
            llm_configs = {}

        # 构建AgentActivityConfigObjeto
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # SiLLM没TenerGenerar，使用ReglaGenerar
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0),
            )
            configs.append(config)

        return configs

    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """基于ReglaGenerar单ElementosAgentConfiguración（中国人作息）"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            # 官方机构：工作TiempoCampaña，低频率，高Influencia
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0,
            }
        elif entity_type in ["mediaoutlet"]:
            # 媒体：全天Campaña，中等频率，高Influencia
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5,
            }
        elif entity_type in ["proFessor", "expert", "official"]:
            # 专家/教授：工作+晚间Campaña，中等频率
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0,
            }
        elif entity_type in ["student"]:
            # 学生：晚间为主，高频率
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                ],  # 上午+晚间
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8,
            }
        elif entity_type in ["alumni"]:
            # 校友：晚间为主
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # 午休+晚间
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
        else:
            # 普通人：晚间高峰
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [
                    9,
                    10,
                    11,
                    12,
                    13,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                ],  # 白天+晚间
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
