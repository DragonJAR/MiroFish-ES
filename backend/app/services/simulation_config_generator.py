"""
Generador inteligente de configuración de simulación
Usa LLM para generar automáticamente parámetros de simulación detallados según requisito de simulación, contenido de documentos e información de grafo
Implementa automatización completa, ninguno necesita configuración manual de parámetros

Adoptar estrategia de generación por pasos para evitar fallido por generar contenido demasiado largo de una vez:
1. Generar configuración de tiempo
2. Generar configuración de eventos
3. Generar configuración de agente en lotes
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
from ..utils.locale import get_language_instruction, t
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger("mirofish.simulation_config")


def _safe_int(value: Any, default: int) -> int:
    """Parsea valor a int de forma segura. Retorna default si no es convertible."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
    logger.warning(f"Invalid int value '{value}', using default {default}")
    return default


def _safe_int_list(value: Any, default: List[int]) -> List[int]:
    """Parsea valor a list[int] de forma segura. Retorna default si no es convertible."""
    if not isinstance(value, list):
        logger.warning(f"Invalid int list '{value}', using default {default}")
        return default
    result = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            result.append(item)
        elif isinstance(item, str):
            try:
                result.append(int(item))
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-parseable int item '{item}'")
        else:
            logger.warning(f"Skipping invalid int list item type {type(item).__name__}")
    if not result:
        logger.warning(f"Empty int list after filtering, using default {default}")
        return default
    return result


# Configuración de tiempo de hábitos chinos (hora de Beijing)
CHINA_TIMEZONE_CONFIG = {
    # Horas de madrugada (casi ninguna actividad)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Horario matutino (despertando gradualmente)
    "morning_hours": [6, 7, 8],
    # Horario laboral
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Horas pico nocturnas (más activas)
    "peak_hours": [19, 20, 21, 22],
    # Horas nocturnas (nivel de actividad descendiendo)
    "night_hours": [23],
    # Coeficientes de nivel de actividad
    "activity_multipliers": {
        "dead": 0.05,  # Madrugada casi ninguna actividad
        "morning": 0.4,  # Mañana gradualmente activo
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

    # Configuración de nivel de actividad (0.0-1.0)
    activity_level: float = 0.5  # Nivel de actividad general

    # Frecuencia de publicaciones (número esperado de publicaciones por hora)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Período de actividad (sistema de 24 horas, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Velocidad de respuesta (retraso de reacción a eventos candentes, unidad: minutos de simulación)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Tendencia emocional (-1.0 a 1.0, negativo a positivo)
    sentiment_bias: float = 0.0

    # Postura (actitud hacia temas específicos)
    stance: str = "neutral"  # supportive, opposing, neutral, observer

    # Peso de influencia (determina la probabilidad de que sus publicaciones sean vistas por otros agentes)
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Configuración de simulación de tiempo (basada en hábitos chinos)"""

    # Duración total de simulación (número de horas de simulación)
    total_simulation_hours: int = 72  # Por defecto simular 72 horas (3 días)

    # Tiempo representado por ronda (minutos de simulación) - Por defecto 60 minutos (1 hora), acelerar flujo de tiempo
    minutes_per_round: int = 60

    # Rango de cantidad de agentes activados por hora
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Horas pico (19-22h, tiempo más activo para chinos)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Horas valle (0-5h, casi ninguna actividad)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = (
        0.05  # Nivel de actividad extremadamente bajo en madrugada
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

    # Eventos programados (eventos desencadenados en tiempos específicos)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Palabras clave de temas candentes
    hot_topics: List[str] = field(default_factory=list)

    # Dirección de guía de opinión pública
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Configuración específica de plataforma"""

    platform: str  # twitter or reddit

    # Peso de algoritmo de recomendación
    recency_weight: float = 0.4  # Frescura de tiempo
    popularity_weight: float = 0.3  # Popularidad
    relevance_weight: float = 0.3  # Relevancia

    # Umbral de propagación viral (cuánta interacción antes de disparar difusión)
    viral_threshold: int = 10

    # Intensidad del efecto de cámara de eco (grado de concentración de opiniones similares)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Configuración completa de parámetros de simulación"""

    # Información base
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Configuración de tiempo
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Lista de configuración de agentes
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Configuración de eventos
    event_config: EventConfig = field(default_factory=EventConfig)

    # Configuración de plataforma
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # Configuración de LLM
    llm_model: str = ""
    llm_base_url: str = ""

    # Metadatos de generación
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # Explicación de razonamiento del LLM

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
        """Convertir a cadena JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Generador inteligente de configuración de simulación

    Usa LLM para analizar requisito de simulación, contenido de documentos, información de entidades del grafo,
    y generar automáticamente la mejor configuración de parámetros de simulación

    Adopta estrategia de generación por pasos:
    1. Generar configuración de tiempo y configuración de eventos (ligero)
    2. Generar configuración de agentes en lotes (cada lote de 10-20 elementos)
    3. Generar configuración de plataforma
    """

    # Máximo número de caracteres de contexto
    MAX_CONTEXT_LENGTH = 50000
    # Cantidad de agentes por lote
    AGENTS_PER_BATCH = 15

    # Longitud de truncamiento de contexto por paso (número de caracteres)
    TIME_CONFIG_CONTEXT_LENGTH = 10000  # Configuración de tiempo
    EVENT_CONFIG_CONTEXT_LENGTH = 8000  # Configuración de eventos
    ENTITY_SUMMARY_LENGTH = 300  # Resumen de entidad
    AGENT_SUMMARY_LENGTH = 300  # Resumen de entidad en configuración de agente
    ENTITIES_PER_TYPE_DISPLAY = 20  # Cantidad de entidades por tipo a mostrar

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
            raise ValueError("LLM_API_KEY no configurada")

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
        Generar inteligentemente configuración completa de simulación (por pasos)

        Args:
            simulation_id: ID de simulación
            project_id: ID de proyecto
            graph_id: ID de grafo
            simulation_requirement: Descripción de requisito de simulación
            document_text: Contenido de documentación original
            entities: Lista de entidades filtradas
            enable_twitter: Si habilitar Twitter
            enable_reddit: Si habilitar Reddit
            progress_callback: Función de callback de progreso (current_step, total_steps, message)

        Returns:
            SimulationParameters: Parámetros completos de simulación
        """
        logger.info(
            f"Inicio de generación inteligente de configuración de simulación: simulation_id={simulation_id}, cantidad de entidades={len(entities)}"
        )

        # Calcular número total de pasos
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = (
            3 + num_batches
        )  # Configuración de tiempo + Configuración de eventos + N lotes de agentes + Configuración de plataforma
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. Construir contexto base
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
        )

        reasoning_parts = []

        # ========== Paso 1: Generar configuración de tiempo ==========
        report_progress(1, t("progress.generatingTimeConfig"))
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(
            f"{t('progress.timeConfigLabel')}: {time_config_result.get('reasoning', t('common.success'))}"
        )

        # ========== Paso 2: Generar configuración de eventos ==========
        report_progress(2, t("progress.generatingEventConfig"))
        event_config_result = self._generate_event_config(
            context, simulation_requirement, entities
        )
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(
            f"{t('progress.eventConfigLabel')}: {event_config_result.get('reasoning', t('common.success'))}"
        )

        # ========== Paso 3-N: Generar configuración de agentes en lotes ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                t(
                    "progress.generatingAgentConfig",
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

        # ========== Asignar agente publicador a publicaciones iniciales ==========
        logger.info(
            "Asignando agente publicador apropiado a publicaciones iniciales..."
        )
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len(
            [
                p
                for p in event_config.initial_posts
                if p.get("poster_agent_id") is not None
            ]
        )
        reasoning_parts.append(t("progress.postAssignResult", count=assigned_count))

        # ========== Último paso: Generar configuración de plataforma ==========
        report_progress(total_steps, t("progress.generatingPlatformConfig"))
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
            f"Generación de configuración de simulación completada: {len(params.agent_configs)} elementos de configuración de agente"
        )

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
    ) -> str:
        """Construir contexto LLM, truncar hasta longitud máxima"""

        # Resumen de entidades
        entity_summary = self._summarize_entities(entities)

        # Construir contexto
        context_parts = [
            f"## Requisito de simulación\n{simulation_requirement}",
            f"\n## Información de entidades ({len(entities)} elementos)\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = (
            self.MAX_CONTEXT_LENGTH - current_length - 500
        )  # Dejar margen de 500 caracteres

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(documentación truncada)"
            context_parts.append(
                f"\n## Contenido de documentación original\n{doc_text}"
            )

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
            lines.append(f"\n### {entity_type} ({len(type_entities)} elementos)")
            # Usar cantidad de muestra y longitud de resumen de la configuración
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
                    f"  ... Aún quedan {len(type_entities) - display_count} elementos"
                )

        return "\n".join(lines)

    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Llamada LLM con reintento, contiene lógica de corrección JSON"""
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
                    - (attempt * 0.1),  # Reducir temperatura en cada reintento
                    # No configurar max_tokens, dejar que LLM decida
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Verificar si fue truncado
                if finish_reason == "length":
                    logger.warning(f"Salida LLM truncada (intento {attempt + 1})")
                    content = self._fix_truncated_json(content)

                # Intentar analizar JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Análisis JSON fallido (intento {attempt + 1}): {str(e)[:80]}"
                    )

                    # Intentar corregir JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(
                    f"Llamada LLM fallida (intento {attempt + 1}): {str(e)[:80]}"
                )
                last_error = e
                import time

                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("Llamada LLM fallida")

    def _fix_truncated_json(self, content: str) -> str:
        """Corregir JSON truncado"""
        content = content.strip()

        # Calcular llaves sin cerrar
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Verificar si hay cadenas sin cerrar
        if content and content[-1] not in '",}]':
            content += '"'

        # Cerrar llaves
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Intentar corregir configuración JSON"""
        import re

        # Corregir casos truncados
        content = self._fix_truncated_json(content)

        # Extraer parte JSON
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # Eliminar saltos de línea en cadenas
            def fix_string(match):
                s = match.group(0)
                s = s.replace("\n", " ").replace("\r", " ")
                s = re.sub(r"\s+", " ", s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError):
                # Intentar eliminar todos los caracteres de control
                json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                json_str = re.sub(r"\s+", " ", json_str)
                try:
                    return json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    pass

        return None

    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Generar configuración de tiempo"""
        # Usar longitud de truncamiento de contexto de la configuración
        context_truncated = context[: self.TIME_CONFIG_CONTEXT_LENGTH]

        # Calcular valor máximo permitido (90% del número de agentes)
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Basado en el siguiente requisito de simulación, generar configuración de simulación de tiempo.

{context_truncated}

## Tarea
Por favor generar configuración de tiempo en JSON.

### Principios básicos (solo para referencia, ajustar según evento específico y grupo de participantes):
- Por favor inferir zona horaria y hábitos de sueño/vigilia del grupo de usuarios objetivo según escenario de simulación, ejemplo de referencia para zona UTC+8
- Madrugada 0-5 Casi nadie en campaña (coeficiente nivel de actividad 0.05)
- Mañana 6-8 gradualmente activo (coeficiente nivel de actividad 0.4)
- Horario de trabajo 9-18 actividad media (coeficiente nivel de actividad 0.7)
- Noche 19-22 es horario pico (coeficiente nivel de actividad 1.5)
- Después de 23h actividad baja (coeficiente nivel de actividad 0.5)
- Regla general: madrugada baja actividad, mañana aumenta gradualmente, trabajo medio, noche pico
- **Importante**: los valores de ejemplo solo son para referencia, necesitas ajustar horarios específicos según naturaleza del evento y características del grupo de participantes
  - Por ejemplo: grupo estudiantil horario pico puede ser 21-23; medios activos todo el día; instituciones oficiales solo en horario laboral
  - Por ejemplo: tema candente repentino puede causar discusión incluso de noche, off_peak_hours puede acortarse apropiadamente

### Formato de retorno JSON (sin markdown)

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
    "reasoning": "Explicación de configuración para este evento"
}}

Descripción de campos:
- total_simulation_hours (int): Duración total de simulación, 24-168 horas, evento repentino corto, tema continuo largo
- minutes_per_round (int): Duración por ronda, 30-120 minutos, sugerencia 60 minutos
- agents_per_hour_min (int): Mínimo de agentes activados por hora (rango: 1-{max_agents_allowed})
- agents_per_hour_max (int): Máximo de agentes activados por hora (rango: 1-{max_agents_allowed})
- peak_hours (arreglo de int): Horas pico, ajustar según evento y grupo de participantes
- off_peak_hours (arreglo de int): Horas valle, generalmente madrugada
- morning_hours (arreglo de int): Horario matutino
- work_hours (arreglo de int): Horario laboral
- reasoning (string): Explicación breve de por qué esta configuración"""

        system_prompt = "Eres experto en simulación de redes sociales. Retornar en formato JSON puro, la configuración de tiempo debe coincidir con los hábitos de sueño/vigilia del grupo de usuarios objetivo en el escenario de simulación."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(
                f"Generación de configuración de tiempo con LLM fallida: {e}, usando configuración por defecto"
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
            "reasoning": "Usando configuración de hábitos chinos por defecto (cada ronda 1 hora)",
        }

    def _parse_time_config(
        self, result: Dict[str, Any], num_entities: int
    ) -> TimeSimulationConfig:
        """Analizar resultado de configuración de tiempo, y verificar que valores de agents_per_hour no excedan total de agentes"""
        # Obtener valores originales
        agents_per_hour_min = result.get(
            "agents_per_hour_min", max(1, num_entities // 15)
        )
        agents_per_hour_max = result.get(
            "agents_per_hour_max", max(5, num_entities // 5)
        )

        # Verificar y corregir: asegurar no exceder total de agentes
        if agents_per_hour_min > num_entities:
            logger.warning(
                f"agents_per_hour_min ({agents_per_hour_min}) excede total de agentes ({num_entities}), corregido"
            )
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(
                f"agents_per_hour_max ({agents_per_hour_max}) excede total de agentes ({num_entities}), corregido"
            )
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # Asegurar min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(
                f"agents_per_hour_min >= max, corregido a {agents_per_hour_min}"
            )

        return TimeSimulationConfig(
            total_simulation_hours=_safe_int(result.get("total_simulation_hours"), 72),
            minutes_per_round=_safe_int(result.get("minutes_per_round"), 60),
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=_safe_int_list(result.get("peak_hours"), [19, 20, 21, 22]),
            off_peak_hours=_safe_int_list(
                result.get("off_peak_hours"), [0, 1, 2, 3, 4, 5]
            ),
            off_peak_activity_multiplier=0.05,  # Madrugada casi nadie
            morning_hours=_safe_int_list(result.get("morning_hours"), [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=_safe_int_list(result.get("work_hours"), list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5,
        )

    def _generate_event_config(
        self, context: str, simulation_requirement: str, entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Generar configuración de eventos"""

        # Obtener lista de tipos de entidad disponibles, para referencia del LLM
        entity_types_available = list(
            set(e.get_entity_type() or "Unknown" for e in entities)
        )

        # Listar nombre de entidad representativo para cada tipo
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

        # Usar longitud de truncamiento de contexto de la configuración
        context_truncated = context[: self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Basado en el siguiente requisito de simulación, generar configuración de eventos.

Requisito de simulación: {simulation_requirement}

{context_truncated}

## Tipos de entidad disponibles y ejemplos
{type_info}

## Tarea
Por favor generar configuración de eventos en JSON:
- Extraer palabras clave de temas candentes
- Describir dirección de desarrollo de opinión pública
- Diseñar contenido de publicaciones iniciales, **para cada publicación se debe especificar poster_type (tipo de publicador)**

**Importante**: poster_type debe seleccionarse de los "tipos de entidad disponibles" de arriba, para que las publicaciones iniciales puedan asignarse al agente publicador apropiado.
Por ejemplo: declaración oficial debe publicarse por tipo Official/University, noticias por MediaOutlet, opiniones de estudiantes por Student.

Retornar formato JSON (sin markdown):
{{
    "hot_topics": ["palabra clave 1", "palabra clave 2", ...],
    "narrative_direction": "<descripción de dirección de desarrollo de opinión pública>",
    "initial_posts": [
        {{"content": "contenido de publicación", "poster_type": "Tipo de entidad (debe seleccionarse de tipos disponibles)"}},
        ...
    ],
    "reasoning": "<explicación breve>"
}}"""

        system_prompt = "Eres experto en análisis de opinión pública. Retornar en formato JSON puro. Nota: el campo 'poster_type' debe coincidir exactamente con los tipos de entidad disponibles."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'poster_type' field value MUST be in English PascalCase exactly Matching the available entity types. Only 'content', 'narrative_direction', 'hot_topics' and 'reasoning' fields should use the specified language."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(
                f"Generación de configuración de eventos con LLM fallida: {e}, usando configuración por defecto"
            )
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Usando configuración por defecto",
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Analizar resultado de configuración de eventos"""
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
        Asignar agente publicador apropiado a publicaciones iniciales

        Basado en el poster_type de cada publicación, encontrar el agent_id más apropiado
        """
        if not event_config.initial_posts:
            return event_config

        # Crear índice de agentes por tipo de entidad
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # Tabla de mapeo de tipos (para manejar diferentes formatos que LLM puede devolver)
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

        # Registrar índice de agente usado para cadaclase de tipo, para evitar usar el mismo agente repetidamente
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            # Intentar encontrar agente coincidente
            matched_agent_id = None

            # 1. Coincidencia directa
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Usar coincidencia por alias
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

            # 3. Si aún no se encuentra, usar agente con mayor influencia
            if matched_agent_id is None:
                logger.warning(
                    f"No se encontró agente para tipo '{poster_type}', usando agente de mayor influencia"
                )
                if agent_configs:
                    # Ordenar por influencia, seleccionar el de mayor
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
                f"Asignación de publicación inicial: poster_type='{poster_type}' -> agent_id={matched_agent_id}"
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
        """Generar configuración de agentes en lotes"""

        # Construir información de entidades (usando longitud de resumen de la configuración)
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

        prompt = f"""Basado en la siguiente información, generar configuración de campaña en redes sociales para cada entidad.

Requisito de simulación: {simulation_requirement}

## Lista de entidades
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Tarea
Generar configuración de campaña para cada entidad, notar:
- **Horario debe coincidir con hábitos del grupo de usuarios objetivo**: a continuación referencia (zona UTC+8), por favor ajustar según escenario de simulación
- **Instituciones oficiales** (University/GovernmentAgency): nivel de actividad bajo (0.1-0.3), campaña en horario laboral (9-17), respuesta lenta (60-240 minutos), influencia alta (2.5-3.0)
- **Medios** (MediaOutlet): nivel de actividad medio (0.4-0.6), campaña todo el día (8-23), respuesta rápida (5-30 minutos), influencia alta (2.0-2.5)
- **Individuos** (Student/Person/Alumni): nivel de actividad alto (0.6-0.9), principalmente campaña nocturna (18-23), respuesta rápida (1-15 minutos), influencia baja (0.8-1.2)
- **Personas públicas/expertos**: nivel de actividad medio (0.4-0.6), influencia media-alta (1.5-2.0)

Retornar formato JSON (sin markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <debe coincidir con entrada>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <frecuencia de publicación>,
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

        system_prompt = "Eres experto en análisis de comportamiento en redes sociales. Retornar en JSON puro, la configuración debe coincidir con los hábitos de sueño/vigilia del grupo de usuarios objetivo en el escenario de simulación."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'stance' field value MUST be one of the English strings: 'supportive', 'opposing', 'neutral', 'observer'. All JSON field names and numeric values must remain unchanged. Only natural language text fields should use the specified language."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {
                cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])
            }
        except Exception as e:
            logger.warning(
                f"Generación de lote de configuración de agentes con LLM fallida: {e}, usando generación por reglas"
            )
            llm_configs = {}

        # Construir objeto AgentActivityConfig
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # Si LLM no generó, usar generación por reglas
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
        """Generar configuración de un solo agente por reglas (hábitos chinos)"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Institución oficial: campaña en horario laboral, baja frecuencia, alta influencia
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
            # Medios: campaña todo el día, frecuencia media, alta influencia
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
            # Experto/profesor: campaña en horario laboral más noche, frecuencia media
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
                ],  # Mañana más noche
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8,
            }
        elif entity_type in ["alumni"]:
            # Egresado: principalmente noche
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # Mediodía más noche
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
                ],  # Día más noche
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0,
            }
