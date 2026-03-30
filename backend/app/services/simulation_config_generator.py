"""
Generador inteligente de configuración de simulación
Utiliza LLM para generar automáticamente parámetros de simulación detallados según los requisitos,
el contenido del documento y la información del grafo. Implementa automatización completa sin necesidad de configuración manual.

Utiliza una estrategia de generación por pasos para evitar fallos por generar demasiado contenido de una vez:
1. Generar configuración de tiempo
2. Generar configuración de eventos
3. Generar configuración de Agentes por lotes
4. Generar configuración de plataforma
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..memory import EntityNode

# Intentar importar el sistema de prompts i18n (fallback a strings vacíos si no existe)
try:
    from ..prompts import load_prompt as _load_prompt

    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False
    _load_prompt = None

logger = get_logger("mirofish.simulation_config")

# Configuración de zona horaria china (hora de Beijing)
CHINA_TIMEZONE_CONFIG = {
    # Horario nocturno (casi nadie activo)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Horario matutino (despertando gradualmente)
    "morning_hours": [6, 7, 8],
    # Horario laboral
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Hora pico nocturno (más activo)
    "peak_hours": [19, 20, 21, 22],
    # Horario nocturno (actividad decrece)
    "night_hours": [23],
    # Coeficientes de actividad
    "activity_multipliers": {
        "dead": 0.05,  # Madrugada casi nadie
        "morning": 0.4,  # Mañana gradualmente activo
        "work": 0.7,  # Horario laboral moderado
        "peak": 1.5,  # Hora pico nocturno
        "night": 0.5,  # Noche decrece
    },
}


@dataclass
class AgentActivityConfig:
    """Configuración de actividad de un solo Agent"""

    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Configuración de actividad (0.0-1.0)
    activity_level: float = 0.5  # Actividad general

    # Frecuencia de publicación (publicaciones esperadas por hora)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Horario activo (formato 24 horas, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Velocidad de respuesta (retraso de reacción a eventos candentes, en minutos de simulación)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Tendencia de sentimiento (-1.0 a 1.0, negativo a positivo)
    sentiment_bias: float = 0.0

    # Posición (actitud hacia temas específicos)
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Peso de influencia (determina probabilidad de que sus publicaciones sean vistas por otros Agents)
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Configuración de simulación de tiempo (basada en hábitos chinos)"""

    # Duración total de la simulación (horas de simulación)
    total_simulation_hours: int = 72  # Por defecto 72 horas (3 días)

    # Tiempo por ronda (minutos de simulación) - Por defecto 60 minutos (1 hora), acelera el flujo del tiempo
    minutes_per_round: int = 60

    # Rango de Agents activados por hora
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Horas pico (19-22, horario de mayor actividad)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Horas valle (0-5 de madrugada, casi nadie activo)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Madrugada actividad muy baja

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

    # Eventos iniciales (eventos activados al inicio de la simulación)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Eventos programados (eventos activados en momentos específicos)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Palabras clave de temas candentes
    hot_topics: List[str] = field(default_factory=list)

    # Dirección de la narrativa
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Configuración específica de plataforma"""

    platform: str  # twitter or reddit

    # Pesos del algoritmo de recomendación
    recency_weight: float = 0.4  # Frescura del tiempo
    popularity_weight: float = 0.3  # Popularidad
    relevance_weight: float = 0.3  # Relevancia

    # Umbral de viralización (interacciones necesarias para activar difusión)
    viral_threshold: int = 10

    # Intensidad del efecto de cámara de eco (agrupación de opiniones similares)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Configuración completa de parámetros de simulación"""

    # Información básica
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Configuración de tiempo
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Lista de configuración de Agents
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Configuración de eventos
    event_config: EventConfig = field(default_factory=EventConfig)

    # Configuración de plataforma
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # Configuración LLM
    llm_model: str = ""
    llm_base_url: str = ""

    # Metadatos de generación
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # Razonamiento del LLM

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
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
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convertir a string JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Generador inteligente de configuración de simulación

    Utiliza LLM para analizar requisitos de simulación, contenido de documentos,
    información de entidades del grafo, y generar automáticamente la mejor configuración de parámetros

    Utiliza estrategia de generación por pasos:
    1. Generar configuración de tiempo y eventos (ligero)
    2. Generar configuración de Agents por lotes (10-20 por lote)
    3. Generar configuración de plataforma
    """

    # Máximo de caracteres del contexto
    MAX_CONTEXT_LENGTH = 50000
    # Número de Agents por lote
    AGENTS_PER_BATCH = 15

    # Longitud de truncamiento del contexto por paso (caracteres)
    TIME_CONFIG_CONTEXT_LENGTH = 10000  # Configuración de tiempo
    EVENT_CONFIG_CONTEXT_LENGTH = 8000  # Configuración de eventos
    ENTITY_SUMMARY_LENGTH = 300  # Resumen de entidad
    AGENT_SUMMARY_LENGTH = 300  # Resumen de entidad en configuración de Agent
    ENTITIES_PER_TYPE_DISPLAY = 20  # Entidades mostradas por tipo

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
            raise ValueError("LLM_API_KEY no configurado")

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
        Generar inteligentemente la configuración completa de simulación (generación por pasos)

        Args:
            simulation_id: ID de simulación
            project_id: ID de proyecto
            graph_id: ID del grafo
            simulation_requirement: Descripción de requisitos de simulación
            document_text: Contenido original del documento
            entities: Lista de entidades filtradas
            enable_twitter: Si habilitar Twitter
            enable_reddit: Si habilitar Reddit
            progress_callback: Función de callback de progreso(current_step, total_steps, message)

        Returns:
            SimulationParameters: Parámetros completos de simulación
        """
        logger.info(
            f"Iniciando generación inteligente de configuración: simulation_id={simulation_id}, entidades={len(entities)}"
        )

        # Calcular número total de pasos
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = (
            3 + num_batches
        )  # Tiempo + eventos + N lotes de Agents + plataforma
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. Construir información de contexto base
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
        )

        reasoning_parts = []

        # ========== Paso 1: Generar configuración de tiempo ==========
        report_progress(1, "Generando configuración de tiempo...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(
            f"Configuración de tiempo: {time_config_result.get('reasoning', 'Éxito')}"
        )

        # ========== Paso 2: Generar configuración de eventos ==========
        report_progress(2, "Generando configuración de eventos y temas candentes...")
        event_config_result = self._generate_event_config(
            context, simulation_requirement, entities
        )
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(
            f"Configuración de eventos: {event_config_result.get('reasoning', 'Éxito')}"
        )

        # ========== Pasos 3-N: Generar configuración de Agents por lotes ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                f"Generando configuración de Agent ({start_idx + 1}-{end_idx}/{len(entities)})...",
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement,
            )
            all_agent_configs.extend(batch_configs)

        reasoning_parts.append(
            f"Configuración de Agent: Generados {len(all_agent_configs)} con éxito"
        )

        # ========== Asignar Agents publicadores a posts iniciales ==========
        logger.info("Asignando Agents publicadores a posts iniciales...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len(
            [
                p
                for p in event_config.initial_posts
                if p.get("poster_agent_id") is not None
            ]
        )
        reasoning_parts.append(
            f"Asignación de posts iniciales: {assigned_count} posts asignados"
        )

        # ========== Último paso: Generar configuración de plataforma ==========
        report_progress(total_steps, "Generando configuración de plataforma...")
        twitter_config = None
        reddit_config = None

        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5,
            )

        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6,
            )

        # Construir parámetros finales
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
            generation_reasoning=" | ".join(reasoning_parts),
        )

        logger.info(
            f"Generación de configuración completada: {len(params.agent_configs)} configuraciones de Agent"
        )

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
    ) -> str:
        """Construir contexto LLM, truncar a longitud máxima"""

        # Resumen de entidades
        entity_summary = self._summarize_entities(entities)

        # Construir contexto
        context_parts = [
            f"## Requisitos de simulación\n{simulation_requirement}",
            f"\n## Información de entidades ({len(entities)} total)\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = (
            self.MAX_CONTEXT_LENGTH - current_length - 500
        )  # Margen de 500 caracteres

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(Documento truncado)"
            context_parts.append(f"\n## Contenido original del documento\n{doc_text}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Generar resumen de entidades"""
        lines = []

        # Agrupar por tipo
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} total)")
            # Usar cantidad mostrada y longitud de resumen configuradas
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
                lines.append(f"  ... otros {len(type_entities) - display_count} más")

        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Llamada LLM con reintento, incluye lógica de reparación de JSON"""
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
                    response_format={"type": "json_object"},
                    temperature=0.7
                    - (attempt * 0.1),  # Reducir temperatura en cada intento
                    # No establecer max_tokens, dejar que el LLM decida
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Verificar si fue truncado
                if finish_reason == "length":
                    logger.warning(f"LLM output truncado (attempt {attempt + 1})")
                    content = self._fix_truncated_json(content)

                # Intentar parsear JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Error parsing JSON (attempt {attempt + 1}): {str(e)[:80]}"
                    )

                    # Intentar reparar JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(
                    f"Llamada LLM fallida (attempt {attempt + 1}): {str(e)[:80]}"
                )
                last_error = e
                import time

                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("Llamada LLM fallida")

    def _fix_truncated_json(self, content: str) -> str:
        """Reparar JSON truncado"""
        content = content.strip()

        # Calcular paréntesis sin cerrar
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Verificar si hay cadenas sin cerrar
        if content and content[-1] not in '",}]':
            content += '"'

        # Cerrar paréntesis
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Intentar reparar configuración JSON"""
        import re

        # Reparar casos truncados
        content = self._fix_truncated_json(content)

        # Extraer parte JSON
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # Remover saltos de línea en cadenas
            def fix_string(match):
                s = match.group(0)
                s = s.replace("\n", " ").replace("\r", " ")
                s = re.sub(r"\s+", " ", s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # Intentar remover todos los caracteres de control
                json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                json_str = re.sub(r"\s+", " ", json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generar configuración de tiempo"""
        # Usar longitud de truncamiento de contexto configurada
        context_truncated = context[: self.TIME_CONFIG_CONTEXT_LENGTH]

        # Calcular máximo permitido (90% del número de agents)
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Basado en los siguientes requisitos de simulación, generar configuración de simulación de tiempo.

{context_truncated}

## Tarea
Por favor generar JSON de configuración de tiempo.

### Principios básicos (solo referencia, ajustar según eventos y grupo de participantes):
- Grupo de usuarios es chino, debe coincidir con horarios de Beijing
- 0-5 de madrugada casi nadie activo (coeficiente 0.05)
- 6-8 mañana gradualmente activo (coeficiente 0.4)
- 9-18 horario laboral moderadamente activo (coeficiente 0.7)
- 19-22 noche hora pico (coeficiente 1.5)
- 23+ actividad decrece (coeficiente 0.5)
- Patrón general: madrugada baja, mañana aumenta, día moderado, noche pico
- **Importante**: Los valores de ejemplo son solo referencia, ajustar según naturaleza del evento y características del grupo
  - Ejemplo: Estudiantes pico puede ser 21-23; medios activos todo el día; entidades oficiales solo horario laboral
  - Ejemplo: Un tema candente repentino puede causar discusión nocturna, off_peak_hours puede acortarse

### Retornar formato JSON (sin markdown)

Ejemplo:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Explicación de configuración de tiempo para este evento"
}}

Campos:
- total_simulation_hours (int): Duración total, 24-168 horas, eventos cortos, temas largos
- minutes_per_round (int): Duración por ronda, 30-120 minutos, recomendado 60
- agents_per_hour_min (int): Mínimos Agents por hora (rango: 1-{max_agents_allowed})
- agents_per_hour_max (int): Máximos Agents por hora (rango: 1-{max_agents_allowed})
- peak_hours (array int): Horas pico, ajustar según grupo
- off_peak_hours (int array): Horas valle, usualmente madrugada
- morning_hours (int array): Horas matutinas
- work_hours (int array): Horas laborales
- reasoning (string): Explicación breve de por qué se configuró así"""

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            system_prompt = _load_prompt("simulation", "time_config_system")
        else:
            # Fallback al original en chino
            system_prompt = "Eres experto en simulación de redes sociales. Retorna JSON puro, configuración de tiempo debe coincidir con hábitos chinos."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(
                f"Generación LLM de configuración de tiempo fallida: {e}, usando configuración por defecto"
            )
            return self._get_default_time_config(num_entities)

    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Obtener configuración de tiempo por defecto (hábitos chinos)"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # Cada ronda 1 hora, acelerar flujo de tiempo
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Usando configuración por defecto de hábitos chinos (cada ronda 1 hora)",
        }

    def _parse_time_config(
        self, result: Dict[str, Any], num_entities: int
    ) -> TimeSimulationConfig:
        """Parsear resultado de configuración de tiempo y validar que agents_per_hour no exceda total de agents"""
        # Obtener valores originales
        agents_per_hour_min = result.get(
            "agents_per_hour_min", max(1, num_entities // 15)
        )
        agents_per_hour_max = result.get(
            "agents_per_hour_max", max(5, num_entities // 5)
        )

        # Validar y corregir: asegurar no exceder total de agents
        if agents_per_hour_min > num_entities:
            logger.warning(
                f"agents_per_hour_min ({agents_per_hour_min}) excede total de Agents ({num_entities}), corregido"
            )
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(
                f"agents_per_hour_max ({agents_per_hour_max}) excede total de Agents ({num_entities}), corregido"
            )
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # Asegurar min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(
                f"agents_per_hour_min >= max, corregido a {agents_per_hour_min}"
            )

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get(
                "minutes_per_round", 60
            ),  # Por defecto cada ronda 1 hora
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # Madrugada casi nadie
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5,
        )

    def _generate_event_config(
        self, context: str, simulation_requirement: str, entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generar configuración de eventos"""

        # Obtener lista de tipos de entidad disponibles, referencia para LLM
        entity_types_available = list(
            set(e.get_entity_type() or "Unknown" for e in entities)
        )

        # Listar entidades representativas por tipo
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

        # Usar longitud de truncamiento de contexto configurada
        context_truncated = context[: self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Basado en los siguientes requisitos de simulación, generar configuración de eventos.

Requisitos: {simulation_requirement}

{context_truncated}

## Tipos de entidad disponibles y ejemplos
{type_info}

## Tarea
Por favor generar JSON de configuración de eventos:
- Extraer palabras clave de temas candentes
- Describir dirección del desarrollo de la narrativa
- Diseñar contenido de posts iniciales, **cada post debe especificar poster_type (tipo de publicador)**

**Importante**: poster_type debe seleccionarse de los "tipos de entidad disponibles" de arriba, para que los posts iniciales puedan asignarse al Agent adecuado.
Ejemplo: Declaraciones oficiales deben ser publicadas por tipo Official/University, noticias por MediaOutlet, opiniones de estudiantes por Student.

Retornar JSON (sin markdown):
{{
    "hot_topics": ["palabra clave 1", "palabra clave 2", ...],
    "narrative_direction": "<descripción de dirección de narrativa>",
    "initial_posts": [
        {{"content": "contenido del post", "poster_type": "tipo de entidad (debe seleccionarse de tipos disponibles)"}},
        ...
    ],
    "reasoning": "<explicación breve>"
}}"""

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            system_prompt = _load_prompt("simulation", "event_config_system")
        else:
            # Fallback al original en chino
            system_prompt = "Eres experto en análisis de narrativa. Retorna JSON puro. Nota: poster_type debe coincidir exactamente con tipos de entidad disponibles."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(
                f"Generación LLM de configuración de eventos fallida: {e}, usando configuración por defecto"
            )
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Usando configuración por defecto",
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Parsear resultado de configuración de eventos"""
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
        Asignar Agents publicadores apropiados a posts iniciales

        Según el poster_type de cada post, encontrar el agent_id más adecuado
        """
        if not event_config.initial_posts:
            return event_config

        # Indexar agents por tipo de entidad
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # Tabla de mapeo de tipos (manejar diferentes formatos que LLM puede generar)
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        # Registrar índice de agents usados por tipo para evitar duplicar el mismo agent
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            # Intentar encontrar agent coincidente
            matched_agent_id = None

            # 1. Coincidencia directa
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Usar alias
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

            # 3. Si aún no se encuentra, usar agent con mayor influencia
            if matched_agent_id is None:
                logger.warning(
                    f"No se encontró Agent coincidente para tipo '{poster_type}', usando Agent de mayor influencia"
                )
                if agent_configs:
                    # Ordenar por influencia, seleccionar el de mayor influencia
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
                f"Asignación de post inicial: poster_type='{poster_type}' -> agent_id={matched_agent_id}"
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
        """Generar configuración de Agents por lotes"""

        # Construir información de entidades (usar longitud de resumen configurada)
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

        prompt = f"""Basado en la siguiente información, generar configuración de actividad de redes sociales para cada entidad.

Requisitos: {simulation_requirement}

## Lista de entidades
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Tarea
Generar configuración de actividad para cada entidad (nota:)
- **Tiempo debe coincidir con hábitos chinos**: 0-5 madrugada casi inactivo, 19-22 noche más activo
- **Entidades oficiales** (University/GovernmentAgency): actividad baja(0.1-0.3), horario laboral(9-17), respuesta lenta(60-240 min), influencia alta(2.5-3.0)
- **Medios** (MediaOutlet): actividad media(0.4-0.6), activo todo el día(8-23), respuesta rápida(5-30 min), influencia alta(2.0-2.5)
- **Individuos** (Student/Person/Alumni): actividad alta(0.6-0.9), principalmente noche(18-23), respuesta rápida(1-15 min), influencia baja(0.8-1.2)
- **Figuras públicas/expertos**: actividad media(0.4-0.6), influencia media-alta(1.5-2.0)

Retornar JSON (sin markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <debe coincidir con entrada>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <frecuencia de posts>,
            "comments_per_hour": <frecuencia de comentarios>,
            "active_hours": [<lista de horas activas, considerar hábitos chinos>],
            "response_delay_min": <retraso mínimo de respuesta en minutos>,
            "response_delay_max": <retraso máximo de respuesta en minutos>,
            "sentiment_bias": <-1.0 a 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <peso de influencia>
        }},
        ...
    ]
}}"""

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            system_prompt = _load_prompt("simulation", "agent_config_system")
        else:
            # Fallback al original en chino
            system_prompt = "Eres experto en análisis de comportamiento en redes sociales. Retorna JSON puro, configuración debe coincidir con hábitos chinos."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {
                cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])
            }
        except Exception as e:
            logger.warning(
                f"Generación LLM de lote de configuración de Agent fallida: {e}, generando por reglas"
            )
            llm_configs = {}

        # Construir objetos AgentActivityConfig
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # Si LLM no generó, generar por reglas
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
        """Generar configuración de un solo Agent por reglas (hábitos chinos)"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Entidad oficial: horario laboral, baja frecuencia, alta influencia
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
            # Medios: todo el día, frecuencia media, alta influencia
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
        elif entity_type in ["professor", "expert", "official"]:
            # Experto/Profesor: trabajo + noche, frecuencia media
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
            # Estudiante: principalmente noche, alta frecuencia
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
                ],  # Mañana + noche
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8,
            }
        elif entity_type in ["alumni"]:
            # Ex-alumno: principalmente noche
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # Almuerzo + noche
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
        else:
            # Persona común: pico nocturno
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
                ],  # Día + noche
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
