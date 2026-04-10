"""
Servicio de Agente de Reportes
Implementa generación de reportes en Modo ReACT usando LangChain + Zep

Funcionalidades:
1. Genera reportes basados en requisitos de simulación e inFormación del grafo de Zep
2. Primero Planifica la Estructura del índice, luego genera por secciones
3. Cada sección usa Modo multi-turno ReACT de Pensamiento y reflexión
4. Soporta diálogo con el usuario, llamando Herramientas de recuperación de Forma autónoma
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_tools import (
    ZepToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult,
)

logger = get_logger("mirofish.report_agent")


class ReportLogger:
    """
    Registrador detallado del Agente de Reportes

    Genera agent_log.jsonl en la carpeta de reportes, registrando cada acción detallada.
    Cada línea es un Objeto JSON completo con timestamp, tipo de acción, contenido detallado, etc.
    """

    def __init__(self, report_id: str):
        """
        Inicializar el logger

        Args:
            report_id: report_id, para determinar la ruta del archivo de log
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "agent_log.jsonl"
        )
        self.start_time = datetime.now()
        self._ensure_log_file()

    def _ensure_log_file(self):
        """Asegurar que exista el directorio del archivo de log"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _get_elapsed_time(self) -> float:
        """Obtener tiempo transcurrido desde el inicio (segundos)"""
        return (datetime.now() - self.start_time).total_seconds()

    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None,
    ):
        """
        Registrar una entrada de log

        Args:
            action: tipo de acción como 'start', 'tool_call', 'llm_response', 'section_complete', etc.
            stage: etapa actual como 'Planning', 'geneRating', 'completed'
            details: diccionario de contenido detallado, sin truncar
            section_title: título de la sección actual (opcional)
            section_index: índice de la sección actual (opcional)
        """
        log_entry = {
            "timestamp": datetime.now().isoFormat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details,
        }

        # Agregar al archivo JSONL
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Registrar inicio de generación de reporte"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": t("report.taskStarted"),
            },
        )

    def log_Planning_start(self):
        """Registrar inicio de Planificación del Esquema"""
        self.log(
            action="Planning_start",
            stage="Planning",
            details={"message": t("report.PlanningStart")},
        )

    def log_Planning_context(self, context: Dict[str, Any]):
        """Registrar inFormación de contexto durante la Planificación"""
        self.log(
            action="Planning_context",
            stage="Planning",
            details={"message": t("report.FetchSimContext"), "context": context},
        )

    def log_Planning_complete(self, outline_dict: Dict[str, Any]):
        """Registrar fin de Planificación del Esquema"""
        self.log(
            action="Planning_complete",
            stage="Planning",
            details={"message": t("report.PlanningComplete"), "outline": outline_dict},
        )

    def log_section_start(self, section_title: str, section_index: int):
        """Registrar inicio de generación de sección"""
        self.log(
            action="section_start",
            stage="geneRating",
            section_title=section_title,
            section_index=section_index,
            details={"message": t("report.sectionStart", title=section_title)},
        )

    def log_react_thought(
        self, section_title: str, section_index: int, iteRation: int, thought: str
    ):
        """Registrar Proceso de Pensamiento ReACT"""
        self.log(
            action="react_thought",
            stage="geneRating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteRation": iteRation,
                "thought": thought,
                "message": t("report.reactThought", iteRation=iteRation),
            },
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteRation: int,
    ):
        """Registrar Llamada a herramienta"""
        self.log(
            action="tool_call",
            stage="geneRating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteRation": iteRation,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": t("report.toolCall", toolName=tool_name),
            },
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteRation: int,
    ):
        """Registrar Resultado de Llamada a herramienta (contenido completo, sin truncar)"""
        self.log(
            action="tool_result",
            stage="geneRating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteRation": iteRation,
                "tool_name": tool_name,
                "result": result,  # Resultado completo, sin truncar
                "result_length": len(result),
                "message": t("report.toolResult", toolName=tool_name),
            },
        )

    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteRation: int,
        has_tool_calls: bool,
        has_final_answer: bool,
    ):
        """Registrar Respuesta LLM (contenido completo, sin truncar)"""
        self.log(
            action="llm_response",
            stage="geneRating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteRation": iteRation,
                "response": response,  # Respuesta completa, sin truncar
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": t(
                    "report.llmResponse",
                    hasToolCalls=has_tool_calls,
                    hasFinalAnswer=has_final_answer,
                ),
            },
        )

    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int,
    ):
        """Registrar generación de contenido de sección (solo contenido, no significa que toda la sección esté completa)"""
        self.log(
            action="section_content",
            stage="geneRating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # contenido completo, sin truncar
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": t("report.sectionContentDone", title=section_title),
            },
        )

    def log_section_full_complete(
        self, section_title: str, section_index: int, full_content: str
    ):
        """
        Registrar generación de sección completada

        El frontend debe escuchar este log para determinar si una sección está realMente completada y obtener el contenido completo
        """
        self.log(
            action="section_complete",
            stage="geneRating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": t("report.sectionComplete", title=section_title),
            },
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """记录Generación de informe completada"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": t("report.reportComplete"),
            },
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Registrar Error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": t("report.errorOccurred", error=error_message),
            },
        )


class ReportConsoleLogger:
    """
    Registrador de consola del Agente de Reportes

    Escribir logs estilo consola (INFO, WARNING, etc.) en console_log.txt dentro de la carpeta de reportes。
    Estos logs son diFerentes a agent_log.jsonl, son salida de consola en texto Plano。
    """

    def __init__(self, report_id: str):
        """
        Inicializar registrador de consola

        Args:
            report_id: report_id, para determinar la ruta del archivo de log
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, "reports", report_id, "console_log.txt"
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()

    def _ensure_log_file(self):
        """Asegurar que exista el directorio del archivo de log"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)

    def _setup_file_handler(self):
        """Configurar manejador de archivos para escribir logs también en archivo"""
        import logging

        # Crear manejador de archivos
        self._file_handler = logging.FileHandler(
            self.log_file_path, mode="a", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.INFO)

        # Usar Formato simple igual que la consola
        Formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self._file_handler.setFormatter(Formatter)

        # Agregar al logger relacionado con report_agent
        loggers_to_attach = [
            "mirofish.report_agent",
            "mirofish.zep_tools",
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Evitar agregar duplicados
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close(self):
        """Cerrar manejador de archivos y remover del logger"""
        import logging

        if self._file_handler:
            loggers_to_detach = [
                "mirofish.report_agent",
                "mirofish.zep_tools",
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None

    def __del__(self):
        """Asegurar cerrar manejador de archivos en destructores"""
        self.close()


class ReportStatus(str, Enum):
    """Estado del reporte"""

    PENDING = "pending"
    PLANNING = "Planning"
    GENERATING = "geneRating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Sección del reporte"""

    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": self.content}

    def to_markdown(self) -> str:
        """Convertir a Formato Markdown"""
        md = f"## {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Esquema del reporte"""

    title: str
    summary: str
    sections: List[ReportSection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
        }

    def to_markdown(self) -> str:
        """Convertir a Formato Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Reporte completo"""

    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ═══════════════════════════════════════════════════════════════
# Constantes de Plantilla Prompt
# ═══════════════════════════════════════════════════════════════

# ── Descripción de Herramientas ──

TOOL_DESC_INSIGHT_FORGE = """\
[Búsqueda de Perspicacia Profunda - Herramienta de búsqueda potente]
Esta es nuestra potente función de búsqueda, diseñada para análisis Profundo. Esta:
1. Descompone automáticaMente tu pregunta en sub-preguntas múltiples
2. Busca inFormación del grafo de simulación desde múltiples dimensiones
3. Integra Resultados de búsqueda semántica, análisis de entidades y seguimiento de cadenas de relaciones
4. Devuelve el contenido de búsqueda Más completo y Profundo

[Casos de uso]
- Necesita analizar profundaMente algún tema
- Necesita conocer los múltiples aspectos de un evento
- Necesita obtener Material rico para apoyar las secciones del reporte

[Contenido devuelto]
- Hechos originales relacionados (se pueden citar directaMente)
- Entidades principales insight
- Cadena de relacionesAnálisis"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Búsqueda en Amplitud - Obtener vista panorámica]
Esta herramienta se usa para obtener la vista panorámica completa de los Resultados de simulación, especialMente adecuada para entender el Proceso de evolución de eventos. Esta:
1. Obtiene todos los nodos relacionados y relaciones
2. DiFerencia hechos actualMente válidos de hechos históricos/expIrados
3. Te ayuda a entender cómo ha evolucionado la Opinión pública

[Casos de uso]
- Necesita conocer el desarrollo completo de un evento
- Necesita comparar cambios de Opinión en diFerentes etapas
- Necesita obtener inFormación completa de entidades y relaciones

[Contenido devuelto]
- Hechos actualMente válidos (Resultados de simulación Más recientes)
- Hechos históricos/expIrados (registros de evolución)
- Todas las entidades involucradas"""

TOOL_DESC_QUICK_SEARCH = """\
[Búsqueda Simple - Búsqueda rápida]
Herramienta de búsqueda ligera y rápida, adecuada para consultas de inFormación simples y directas.

[Casos de uso]
- Necesita buscar rápidaMente alguna inFormación específica
- Necesita verificar algún hecho específico
- Consulta de inFormación simple

[Contenido devuelto]
- Lista de hechos Más relevantes con la consulta"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Entrevista Profunda - Entrevista real de Agent (doble plataForma)]
Llama a la API de entrevista del entorno de simulación OASIS para realizar entrevistas reales con los Agentes de simulación que están Corriendo！
Esto no es simulación LLM, sino llamar a la interfaz de entrevista real para obtener las Respuestas originales de los Agentes de simulación.
Por deFecto, entrevista simultáneaMente en Twitter y Reddit dos plataFormas, obteniendo puntos de vista Más completos.

Flujo de funcionalidades:
1. Lee automáticaMente el archivo de configuración de personalidades, entendiendo todos los Agentes de simulación
2. Selecciona inteligenteMente los Agentes Más relevantes con el tema de entrevista (ej. estudiantes, Medios, oficiales, etc.)
3. Genera automáticaMente preguntas de entrevista
4. Llama a la interfaz /api/simulation/interview/batch para realizar entrevistas reales en doble plataForma
5. Integra todos los Resultados de entrevistas, proporcionando análisis multi-Perspectiva

[Casos de uso]
- Necesita conocer puntos de vista desde diFerentes roles sobre un evento (¿qué opinan los estudiantes? ¿qué opinan los Medios? ¿qué dice lo oficial?)
- Necesita recopilar opiniones y posiciones de múltiples Partes
- Necesita obtener Respuestas reales de Agentes de simulación (del entorno de simulación OASIS)
- Quiere que el reporte sea Más Vívido, incluyendo "transcripciones de entrevistas"

[Contenido devuelto]
- InFormación de identidad de Agentes entrevistados
- Respuestas de entrevista en Twitter y Reddit dos plataFormas
- Citas clave (se pueden citar directaMente)
- Resumen de entrevistas y comparación de puntos de vista

[Importante] ¡Necesita que el entorno de simulación OASIS esté Corriendo para usar esta funcionalidad!"""

# ── 大纲规划 prompt ──

PLAN_SYSTEM_PROMPT = """\
Eres un experto en escritura de「Reportes de Predicción del Futuro」y posees una「Perspectiva de Dios» sobre el mundo de simulación — puedes comprender el comportamiento, discursos e interacciones de cada Agente en la simulación.

【Concepto clave】
Hemos construido un mundo de simulación e inyectado un「Requisito de simulación」específico como variable. El Resultado de evolución del mundo de simulación es la predicción de situaciones futuras que pueden ocurrir. No estás observando「datos experimentales」, sino「ensayo del futuro」。

【Tu tarea】
Escribe un「Reporte de Predicción del Futuro」que responda:
1. ¿Qué pasará en el futuro Bajo nuestras condiciones establecidas?
2. ¿Cómo reaccionarán y actuarán los diversos Agentes (multitudes)?
3. ¿Qué tendencias futuras y riesgos merecen atención esta simulación?

【Posicionamiento del reporte】
- ✅ Este es un reporte de predicción del futuro basado en simulación, revelando「si esto sucede, así será el futuro」
- ✅ Enfocado en Resultados de predicción: dirección de eventos, reacciones grupales, FenóMenos emergentes, riesgos potenciales
- ✅ Los actos y dichos de los Agentes en el mundo de simulación son predicciones del comportamiento futuro de las multitudes
- ❌ No es un análisis del estado actual del mundo real
- ❌ No es una revisión general de Opinión pública sin profundidad

【Límite de cantidad de secciones】
- Mínimo 2 secciones, máximo 5 secciones
- No se requieren subsecciones, cada sección escrita directaMente con contenido completo
- El contenido debe ser conciso, enfocado en descubrimientos de predicción clave
- La Estructura de secciones es diseñada por ti según los Resultados de predicción

Por favor, salida el Esquema del reporte en Formato JSON, con el siguiente Formato:
{
    "title": "Título del reporte",
    "summary": "Resumen del reporte (una frase que resume los descubrimientos clave de predicción)",
    "sections": [
        {
            "title": "Título de la sección",
            "description": "Descripción del contenido de la sección"
        }
    ]
}

Nota: el array sections debe tener mínimo 2 Elementos, máximo 5 Elementos！"""

PLAN_USER_PROMPT_TEMPLATE = """\
【Escenario de predicción establecido】
Variables que inyectamos al mundo de simulación (Requisito de simulación): {simulation_requirement}

【Escalas del mundo de simulación】
- Cantidad de entidades participantes en la simulación: {total_nodes}
- Cantidad de relaciones generadas entre entidades: {total_edges}
- Distribución de tipos de entidades: {entity_types}
- Cantidad de Agentes activos: {total_entities}

【Muestra de algunos hechos futuros predichos por la simulación】
{reLated_facts_json}

Por favor, examine este ensayo del futuro desde la「Perspectiva de Dios」:
1. ¿Qué estado se presenta en el futuro Bajo nuestras condiciones establecidas?
2. ¿Cómo reaccionan y actúan los diversos grupos (Agentes)?
3. ¿Qué tendencias futuras merecen atención esta simulación?

Según los Resultados de predicción, diseña la Estructura de secciones Más adecuada para el reporte.

【Recordatorio】Cantidad de secciones del reporte: mínimo 2, máximo 5, contenido conciso enfocado en descubrimientos de predicción clave。"""

# ── 章节Generar prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Eres un experto en escritura de「Reportes de Predicción del Futuro」, escribiendo una sección del reporte.

Título del reporte: {report_title}
Resumen del reporte: {report_summary}
Escenario de predicción (Requisito de simulación): {simulation_requirement}

Sección actual a escribir: {section_title}

════════════════════════════════════════════════════════════
【Concepto clave】
════════════════════════════════════════════════════════════

El mundo de simulación es un ensayo del futuro. Hemos inyectado condiciones específicas (Requisito de simulación) al mundo de simulación,
el comportamiento e interacciones de los Agentes en la simulación son predicciones del comportamiento futuro de las multitudes.

Tu tarea es:
- Revelar qué pasará en el futuro Bajo las condiciones establecidas
- Predecir cómo reaccionarán y actuarán los diversos grupos (Agentes)
- Descubrir tendencias futuras, riesgos y OportUnidades que merecen atención

❌ No lo escribas como un análisis del estado actual del mundo real
✅ Debes enfocarte en「cómo será el futuro」 — los Resultados de simulación son el futuro predicho

════════════════════════════════════════════════════════════
【Reglas Más importantes - Debes cumplir】
══════════════════════════════════════════════════════════════

1. 【Debes llamar Herramientas para observar el mundo de simulación】
   - Estás observando el ensayo del futuro desde la「Perspectiva de Dios」
   - Todo el contenido debe provenir de eventos y actos/dichos de Agentes que ocurren en el mundo de simulación
   - Está prohibido usar tu propio Conocimiento para escribir contenido del reporte
   - Cada sección debe llamar Herramientas al Menos 3 veces (máximo 5 veces) para observar el mundo de simulación, representa el futuro

2. 【Debes citar los actos/dichos originales de Agentes】
   - Los discursos y comportamientos de Agentes son predicciones del comportamiento futuro de las multitudes
   - En el reporte, usa Formato de cita para mostrar estas predicciones, por Ejemplo:
     > "Un cierto grupo expresará: contenido original..."
   - Estas citas son la evidencia central de la predicción de simulación

3. 【Consistencia de idioma - El contenido citado debe traducirse al idioma del reporte】
   - El contenido devuelto por Herramientas puede contener expresiones diFerentes al idioma del reporte
   - El reporte debe escribirse completaMente usando el idioma consistente con el especificado por el usuario
   - Cuando cites contenido devuelto por Herramientas en otro idioma, debes traducirlo al idioma del reporte antes de escribirlo
   - Al traducir, mantén el significado original sin cambios, asegurando que la expresión sea natural y fluida
   - Esta Regla se aplica Tanto al cuerpo principal como al contenido en Bloques de cita (Formato >)

4. 【Presentación fiel de Resultados de predicción】
   - El contenido del reporte debe reflejar Resultados de simulación en el mundo de simulación que representan el futuro
   - No agregar inFormación que no exista en la simulación
   - Si hay inFormación insuficiente en algún aspecto, explícala honestly

════════════════════════════════════════════════════════════
【⚠️ Normas de Formato - ¡Muy importante！】
════════════════════════════════════════════════════════════

【Una sección = Unidad mínima de contenido】
- Cada sección es la Unidad mínima de división del reporte
- ❌ Está prohibido usar cualquier título Markdown (#, ##, ###, ####, etc.) dentro de secciones
- ❌ Está prohibido agregar título de sección al inicio del contenido
- ✅ El título de sección se agrega automáticaMente por el Sistema, solo necesitas escribir contenido en texto Plano
- ✅ Usa **negrita**, separación de párrafos, citas, listas para organizar contenido, pero no uses títulos

【Ejemplo correcto】
```
Esta sección analiza la situación de propagación de Opinión pública. Mediante análisis Profundo de datos de simulación, descubrimos...

**Etapa de detonación inicial**

Weibo como el primer lugar de Opinión pública, asumiendo el rol central de inFormación inicial:

> "Weibo contribuyó con el 68% del volumen inicial de comentarios..."

**Etapa de amplificación emocional**

La plataForma Douyin amplió aún Más la influencia del evento:

- Fuerte impacto visual
- Alta resonancia emocional
```

【Ejemplo de error】
```
## Resumen de ejecución          ← ¡Error! No agregues ningún título
### 1. Primera etapa     ← ¡Error! No uses ### para dividir subsecciones
#### 1.1 Análisis detallado   ← ¡Error! No uses #### para subdividir

Esta sección analiza...
```

══════════════════════════════════════════════════════════════
【Herramientas de recuperación disponibles】(cada sección llama 3-5 veces）
══════════════════════════════════════════════════════════════

{tools_description}

【Sugerencias de uso de Herramientas - Por favor mezcla diFerentes Herramientas, no uses solo una】
- insight_forge: Análisis de perspicacia profunda, descompone automáticaMente preguntas y recupera hechos y relaciones desde múltiples dimensiones
- panorama_search: Búsqueda panorámica en amplitud, entiende vista completa, línea de tiempo y Proceso de evolución del evento
- quick_search: Validación rápida de algún punto específico de inFormación
- interview_agents: Entrevista a Agentes de simulación, obtiene puntos de vista en primera persona y reacciones reales de diFerentes roles

══════════════════════════════════════════════════════════════
【Flujo de traBajo】
════════════════════════════════════════════════════════════════

Cada Respuesta solo puedes hacer una de las siguientes dos Cosas (no simultáneaMente):

Opción A - Llamar herramienta:
Salida tu Pensamiento, luego llama a una herramienta con el siguiente Formato:
<tool_call>
{{"name": "工具Nombre", "parameters": {{"参数名": "参数Valor"}}}}
</tool_call>
Sistema会执Fila工具并把结果 devolver给你。你不Necesita也不能自己编写工具 devolver结果。

OpciónB - 输出Más终Contenido：
Cuando你已A través de工具Obtener了足够Información，以 "Final Answer:" 开头输出章节Contenido。

⚠️ 严格禁止：
- 禁止En一次Responder中同时Contiene工具调用Y Final Answer
- 禁止自己编造工具 devolver 结果（Observation），Todos工具结果由Sistema注入
- 每次ResponderMás多调用 una 工具

════════════════════════════════════════════════════════════
【章节ContenidoRequisito】
══════════════════════════════════════════════════════════════

1. ContenidoDebe基于工具检索Hasta的SimulaciónDatos
2. 大量Citar原文来展示Simulación效果
3. 使用Markdown格式（Pero禁止使用Título）：
   - 使用 **粗体文字** Marcar重点（代替子Título）
   - 使用Lista（-O1.2.3.）Organización要点
   - 使用空Fila分隔不同段落
   - ❌ 禁止使用 #、##、###、#### 等CualquierTítulo语法
4. 【Citar格式Estándar - Debe单独成段】
   CitarDebe独立成段，前后各Tener一空Fila，不能混En段落中：

   ✅ 正确格式：
   ```
   校方的回应被Pensar缺乏实质Contenido。

   > "校方的应Para模式En瞬息万变的社交媒体环境中显得僵化Y迟缓。"

   这一评价反映了公众的普遍不满。
   ```

   ❌ Error 格式：
   ```
   校方的回应被Pensar缺乏实质Contenido。> "校方的应Para模式..." 这一评价反映了...
   ```
5. 保持Con otra 章节的逻辑连贯性
6. 【避免重复】仔细阅读下方 completado 的章节Contenido，不要重复Descripción相同的Información
7. 【再次强调】不要添加CualquierTítulo！用**粗体**代替小节Título"""

SECTION_USER_PROMPT_TEMPLATE = """\
 completado 的章节Contenido（请仔细阅读，避免重复）：
{previous_content}

════════════════════════════════════════════════════════════
【Cuando前Tarea】撰写章节: {section_title}
════════════════════════════════════════════════════════════

【重要Recordatorio】
1. 仔细阅读上方 completado 的章节，避免重复相同的Contenido！
2. Inicio前Debe先调用工具 obtener datos de simulación
3. 请混合使用不同工具，不要Solo用一种
4. 报告ContenidoDebe来自检索结果，不要使用自己的知识

【⚠️ 格式警告 - Debe遵守】
- ❌ 不要写CualquierTítulo（#、##、###、####Todos不Fila）
- ❌ 不要写"{section_title}"Como开头
- ✅ 章节Título由Sistema自动添加
- ✅ 直接写正文，用**粗体**代替小节Título

请Inicio：
1. 首先思考（Thought）这章节NecesitaQuéInformación
2. 然后调用工具（Action）obtener datos de simulación
3. 收集足够Información后输出 Final Answer（纯正文，NingunoCualquierTítulo）"""

# ── ReACT Ciclo内Mensaje模板 ──

REACT_OBSERVATION_TEMPLATE = """\
Observation（检索结果）:

═══ 工具 {tool_name} Volver ═══
{result}

═══════════════════════════════════════════════════════════════
已调用工具 {tool_calls_count}/{max_tool_calls} 次（已用: {used_tools_str}）{unused_hint}
- SiInformación充分：以 "Final Answer:" 开头输出章节Contenido（DebeCitar上述原文）
- SiNecesitaMás多Información：调用一Elementos工具Continuar检索
═══════════════════════════════════════════════════════════════"""


REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Nota】Solo has llamado Herramientas {tool_calls_count} veces, mínimo requiere {min_tool_calls} veces."
    "Por favor llama Herramientas para obtener Más datos de simulación, luego salida Final Answer。{unused_hint}"
)


REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "ActualMente solo has llamado Herramientas {tool_calls_count} veces, mínimo requiere {min_tool_calls} veces。"
    "Por favor llama Herramientas para obtener datos de simulación。{unused_hint}"
)


REACT_TOOL_LIMIT_MSG = (
    "El número de Llamadas a Herramientas ha alcanzado el límite ({tool_calls_count}/{max_tool_calls}), no se pueden llamar Más Herramientas。"
    'Por favor, basándote en la inFormación ya obtenida, salida el contenido de la sección con "Final Answer:" al inicio。'
)


REACT_UNUSED_TOOLS_HINT = "\n💡 Aún no has usado: {unused_list}, se recomienda intentar diFerentes Herramientas para obtener inFormación desde múltiples ángulos"

REACT_FORCE_FINAL_MSG = "Se ha alcanzado el límite de Llamadas a Herramientas, por favor salida directaMente Final Answer: y genera el contenido de la sección。"

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
你Es一Elementos简洁高效的SimulaciónPredicción助手。

【背景】
PredicciónCondición: {simulation_requirement}

【已Generar的Informe de análisis】
{report_content}

【Regla】
1. Prioridad基于上述报告Contenido回答Problema
2. 直接回答Problema，避免冗长的思考论述
3. 仅En报告Contenido不足以回答时，才调用工具检索Más多Datos
4. 回答要简洁、清晰、Tener条理

【可用工具】（仅EnNecesita时使用，Más多调用1-2次）
{tools_description}

【工具调用格式】
<tool_call>
{{"name": "工具Nombre", "parameters": {{"参数名": "参数Valor"}}}}
</tool_call>

【Estilo de Respuesta】
- Conciso y directo, no discursos Largos
- Usa Formato > para citar contenido clave
- Prioriza dar conclusiones, luego explica razones"""

CHAT_OBSERVATION_SUFFIX = "\n\nPor favor responde concisaMente a la pregunta。"


# ═══════════════════════════════════════════════════════════════
# ReportAgent 主Clase
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Agente de Reportes - Generador de Informes

    Adopta el Modo ReACT (Reasoning + Acting):
    1. Fase de Planificación: analiza Requisito de simulación, Planifica la Estructura de directorio del reporte
    2. Fase de generación: genera contenido por secciones, cada sección puede llamar Herramientas múltiples veces para obtener inFormación
    3. Fase de reflexión: verifica integridad y exActitud del contenido
    """

    # Más大工具调用次数（Cada章节）
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Más大反思rondas数
    MAX_REFLECTION_ROUNDS = 3

    # Para话中的Más大工具调用次数
    MAX_TOOL_CALLS_PER_CHAT = 2

    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None,
    ):
        """
        Inicializando Agente de Reportes

        Args:
            graph_id: GrafoID
            simulation_id: SimulaciónID
            simulation_requirement: Descripción de Requisito de simulación
            llm_client: Cliente LLM (opcional)
            zep_tools: Servicio de Herramientas Zep (opcional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()

        # Herramientas definidas
        self.tools = self._define_tools()

        # Registrador de logs (se inicializa en generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Registrador de consola (se inicializa en generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(
            t("report.agentInitDone", graphId=graph_id, simulationId=simulation_id)
        )

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Definir Herramientas disponibles"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Pregunta o tema que deseas analizar Profundo",
                    "report_context": "Contexto de sección actual del reporte (opcional, ayuda a generar sub-preguntas Más precisas)",
                },
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Consulta de búsqueda, para Ordenamiento por relevancia",
                    "include_expired": "Si incluye contenido expIrado/histórico (por deFecto True)",
                },
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Cadena de consulta de búsqueda",
                    "limit": "Cantidad de Resultados a devolver (opcional, por deFecto 10)",
                },
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Tema o descripción de Necesidad de entrevista (ej: 'entender opiniones de estudiantes sobre incidente de Formaldehído en dormitorios')",
                    "max_agents": "Cantidad máxima de Agentes a entrevistar (opcional, por deFecto 5, máximo 10)",
                },
            },
        }

    def _execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], report_context: str = ""
    ) -> str:
        """
        Ejecutar Llamada a herramienta

        Args:
            tool_name: Nombre de herramienta
            parameters: Parámetros de herramienta
            report_context: Contexto del reporte (para InsightForge)

        Returns:
            Resultado de ejecución de herramienta (Formato texto)
        """
        logger.info(t("report.executingTool", toolName=tool_name, params=parameters))

        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx,
                )
                return result.to_text()

            elif tool_name == "panorama_search":
                # Búsqueda en amplitud - obtener vista panorámica
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ["true", "1", "yes"]
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id, query=query, include_expired=include_expired
                )
                return result.to_text()

            elif tool_name == "quick_search":
                # Búsqueda simple - recuperación rápida
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id, query=query, limit=limit
                )
                return result.to_text()

            elif tool_name == "interview_agents":
                # Entrevista profunda - llamar a la API de entrevista OASIS real para obtener Respuestas de Agentes de simulación (doble plataForma)
                interview_topic = parameters.get(
                    "interview_topic", parameters.get("query", "")
                )
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents,
                )
                return result.to_text()

            # ========== Hacia后兼容的旧工具（内部重定HaciaHasta新工具） ==========

            elif tool_name == "search_graph":
                # 重定HaciaHasta quick_search
                logger.info(t("report.redirectToQuickSearch"))
                return self._execute_tool("quick_search", parameters, report_context)

            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id, entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif tool_name == "get_simulation_context":
                # 重定HaciaHasta insight_forge，Porque它Más强大
                logger.info(t("report.redirectToInsightForge"))
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool(
                    "insight_forge", {"query": query}, report_context
                )

            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id, entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)

            else:
                return f"Desconocido工具: {tool_name}。请使用以下工具之一: insight_forge, panorama_search, quick_search"

        except Exception as e:
            logger.error(t("report.toolExecFailed", toolName=tool_name, error=str(e)))
            return f"工具执FilaFallido: {str(e)}"

    # 合法的工具Nombre集合，用于裸 JSON 兜底Analizar时校验
    VALID_TOOL_NAMES = {
        "insight_forge",
        "panorama_search",
        "quick_search",
        "interview_agents",
    }

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        DesdeLLMRespuesta中Analizar工具调用

        Soporte的格式（按Prioridad级）：
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. 裸 JSON（Respuesta整体O单FilaEntoncesEs一Elementos工具调用 JSON）
        """
        tool_calls = []

        # Formato1: XML风格（标准格式）
        xml_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Formato2: 兜底 - LLM 直接输出裸 JSON（没包 <tool_call> Etiqueta）
        # SoloEn格式1未匹配时尝试，避免误匹配正文中的 JSON
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # RespuestaPosibleContiene思考文字 + 裸 JSON，尝试ExtracciónMás后一Elementos JSON Objeto
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """校验Analizar出的 JSON SiEs合法的工具调用"""
        # Soporte {"name": ..., "parameters": ...} Y {"tool": ..., "params": ...} 两种Clave名
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # 统一Clave名为 name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False

    def _get_tools_description(self) -> str:
        """Generar工具Descripción文本"""
        desc_parts = ["可用工具："]
        for name, tool in self.tools.items():
            params_desc = ", ".join(
                [f"{k}: {v}" for k, v in tool["parameters"].items()]
            )
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  参数: {params_desc}")
        return "\n".join(desc_parts)

    def Plan_outline(
        self, progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        规划报告大纲

        使用LLMAnálisisRequisito de simulación，规划报告的Directorio结构

        Args:
            progress_callback: 进度CallbackFunción

        Returns:
            ReportOutline: 报告大纲
        """
        logger.info(t("report.startPlanningOutline"))

        if progress_callback:
            progress_callback("Planning", 0, t("progress.analyzingRequirements"))

        # 首先ObtenerSimulación上下文
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id, simulation_requirement=self.simulation_requirement
        )

        if progress_callback:
            progress_callback("Planning", 30, t("progress.geneRatingOutline"))

        system_prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{get_language_instruction()}"
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.Format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get("graph_statistics", {}).get("total_nodes", 0),
            total_edges=context.get("graph_statistics", {}).get("total_edges", 0),
            entity_types=list(
                context.get("graph_statistics", {}).get("entity_types", {}).keys()
            ),
            total_entities=context.get("total_entities", 0),
            reLated_facts_json=json.dumps(
                context.get("reLated_facts", [])[:10], ensure_ascii=False, indent=2
            ),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            if progress_callback:
                progress_callback("Planning", 80, t("progress.parsingOutline"))

            # Analizar大纲
            sections = []
            for section_data in response.get("sections", []):
                sections.append(
                    ReportSection(title=section_data.get("title", ""), content="")
                )

            outline = ReportOutline(
                title=response.get("title", "SimulaciónInforme de análisis"),
                summary=response.get("summary", ""),
                sections=sections,
            )

            if progress_callback:
                progress_callback("Planning", 100, t("progress.outlinePlanComplete"))

            logger.info(t("report.outlinePlanDone", count=len(sections)))
            return outline

        except Exception as e:
            logger.error(t("report.outlinePlanFailed", error=str(e)))
            # Volver默认大纲（3Elementos章节，Comofallback）
            return ReportOutline(
                title="未来Predicción报告",
                summary="基于SimulaciónPredicción的未来趋势ConRiesgoAnálisis",
                sections=[
                    ReportSection(title="Escenario de predicciónConNúcleoDescubrir"),
                    ReportSection(title="人群Fila为PredicciónAnálisis"),
                    ReportSection(title="趋势展望ConRiesgo提示"),
                ],
            )

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0,
    ) -> str:
        """
        使用ReACT模式Generar单Elementos章节Contenido

        ReACTCiclo：
        1. Thought（思考）- AnálisisNecesitaQuéInformación
        2. Action（Fila动）- 调用工具ObtenerInformación
        3. Observation（观察）- Análisis工具Volver结果
        4. 重复HastaInformación足够O达HastaMás大次数
        5. Final Answer（Más终回答）- Generar章节Contenido

        Args:
            section: 要Generar的章节
            outline: 完整大纲
            previous_sections: Antes章节的Contenido（用于保持连贯性）
            progress_callback: 进度Callback
            section_index: 章节Índice（用于Log记录）

        Returns:
            章节Contenido（Markdown格式）
        """
        logger.info(t("report.reactGenerateSection", title=section.title))

        # Registrar章节InicioLog
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)

        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.Format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # 构建用户prompt - 每ElementosCompletado章节各传入Más大4000字
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # 每Elementos章节Más多4000字
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "（这Es第一Elementos章节）"

        user_prompt = SECTION_USER_PROMPT_TEMPLATE.Format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # ReACTCiclo
        tool_calls_count = 0
        max_iteRations = 5  # Más大Iteraciónrondas数
        min_tool_calls = 3  # Más少工具调用次数
        conflict_retries = 0  # 工具调用ConFinal Answer同时出现的连续冲突次数
        used_tools = set()  # Registrar已调用过的工具名
        all_tools = {
            "insight_forge",
            "panorama_search",
            "quick_search",
            "interview_agents",
        }

        # Reporte上下文，用于InsightForge的子ProblemaGenerar
        report_context = f"章节Título: {section.title}\nRequisito de simulación: {self.simulation_requirement}"

        for iteRation in range(max_iteRations):
            if progress_callback:
                progress_callback(
                    "geneRating",
                    int((iteRation / max_iteRations) * 100),
                    t(
                        "progress.deepSearchAndWrite",
                        current=tool_calls_count,
                        max=self.MAX_TOOL_CALLS_PER_SECTION,
                    ),
                )

            # 调用LLM
            response = self.llm.chat(
                messages=messages, temperature=0.5, max_tokens=4096
            )

            # Inspección LLM VolverSi为 None（API ExcepciónOContenido为空）
            if response is None:
                logger.warning(
                    t(
                        "report.sectionIterNone",
                        title=section.title,
                        iteRation=iteRation + 1,
                    )
                )
                # SiTodavíaTenerIteración次数，添加Mensaje并Reintentar
                if iteRation < max_iteRations - 1:
                    messages.append({"role": "assistant", "content": "（Respuesta为空）"})
                    messages.append({"role": "user", "content": "请ContinuarGenerarContenido。"})
                    continue
                # Más后一次Iteración也Volver None，跳出Ciclo进入强制收尾
                break

            logger.debug(f"LLMRespuesta: {response[:200]}...")

            # Analizar一次，复用结果
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── 冲突Procesar：LLM 同时输出了工具调用Y Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    t(
                        "report.sectionConflict",
                        title=section.title,
                        iteRation=iteRation + 1,
                        conflictCount=conflict_retries,
                    )
                )

                if conflict_retries <= 2:
                    # 前两次：丢弃本次Respuesta，Requisito LLM 重新Responder
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "【格式Error】你En一次Responder中同时Contiene了工具调用Y Final Answer，这Es不允许的。\n"
                                "每次ResponderSolo能Hacer以下两件事之一：\n"
                                "- 调用一Elementos工具（输出一Elementos <tool_call> 块，不要写 Final Answer）\n"
                                "- 输出Más终Contenido（以 'Final Answer:' 开头，不要Contiene <tool_call>）\n"
                                "请重新Responder，SoloHacer其中一件事。"
                            ),
                        }
                    )
                    continue
                else:
                    # 第三次：DegradarProcesar，截断Hasta第一Elementos工具调用，强制执Fila
                    logger.warning(
                        t(
                            "report.sectionConflictDowngrade",
                            title=section.title,
                            conflictCount=conflict_retries,
                        )
                    )
                    first_tool_end = response.find("</tool_call>")
                    if first_tool_end != -1:
                        response = response[: first_tool_end + len("</tool_call>")]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Registrar LLM RespuestaLog
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteRation=iteRation + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer,
                )

            # ── 情况1：LLM 输出了 Final Answer ──
            if has_final_answer:
                # 工具调用次数不足，Rechazo并RequisitoContinuar调工具
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = (
                        f"（Estos工具Todavía未使用，Recomendación用一下他们: {', '.join(unused_tools)}）"
                        if unused_tools
                        else ""
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_INSUFFICIENT_TOOLS_MSG.Format(
                                tool_calls_count=tool_calls_count,
                                min_tool_calls=min_tool_calls,
                                unused_hint=unused_hint,
                            ),
                        }
                    )
                    continue

                # 正常结束
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(
                    t(
                        "report.sectionGenDone",
                        title=section.title,
                        count=tool_calls_count,
                    )
                )

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count,
                    )
                return final_answer

            # ── 情况2：LLM 尝试调用工具 ──
            if has_tool_calls:
                # 工具额度已耗尽 → 明确告知，Requisito输出 Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_TOOL_LIMIT_MSG.Format(
                                tool_calls_count=tool_calls_count,
                                max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            ),
                        }
                    )
                    continue

                # Solo执Fila第一Elementos工具调用
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(
                        t(
                            "report.multiToolOnlyFirst",
                            total=len(tool_calls),
                            toolName=call["name"],
                        )
                    )

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteRation=iteRation + 1,
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context,
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteRation=iteRation + 1,
                    )

                tool_calls_count += 1
                used_tools.add(call["name"])

                # 构建未使用工具提示
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.Format(
                        unused_list="、".join(unused_tools)
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": REACT_OBSERVATION_TEMPLATE.Format(
                            tool_name=call["name"],
                            result=result,
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            used_tools_str=", ".join(used_tools),
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # ── 情况3：既没Tener工具调用，也没Tener Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # 工具调用次数不足，Recomendación未用过的工具
                unused_tools = all_tools - used_tools
                unused_hint = (
                    f"（Estos工具Todavía未使用，Recomendación用一下他们: {', '.join(unused_tools)}）"
                    if unused_tools
                    else ""
                )

                messages.append(
                    {
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.Format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # 工具调用已足够，LLM 输出了ContenidoPero没带 "Final Answer:" 前缀
            # 直接将这段ContenidoComoMás终答案，不再空转
            logger.info(
                t("report.sectionNoPrefix", title=section.title, count=tool_calls_count)
            )
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count,
                )
            return final_answer

        # 达HastaMás大Iteración次数，强制GenerarContenido
        logger.warning(t("report.sectionMaxIter", title=section.title))
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(messages=messages, temperature=0.5, max_tokens=4096)

        # Inspección强制收尾时 LLM VolverSi为 None
        if response is None:
            logger.error(t("report.sectionForceFailed", title=section.title))
            final_answer = t("report.sectionGenFailedContent")
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        # Registrar章节ContenidoGenerarCompletadoLog
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count,
            )

        return final_answer

    def generate_report(
        self,
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None,
    ) -> Report:
        """
        Generar完整报告（分章节实时输出）

        每Elementos章节GenerarCompletado后立即GuardarHastaArchivo夹，不NecesitaPendiente整Elementos报告Completado。
        Archivo结构：
        reports/{report_id}/
            meta.json       - 报告元Información
            outline.json    - 报告大纲
            progress.json   - Generar进度
            section_01.md   - 第1章节
            section_02.md   - 第2章节
            ...
            full_report.md  - 完整报告

        Args:
            progress_callback: 进度CallbackFunción (stage, progress, message)
            report_id: 报告ID（Opcional，Si不传则自动Generar）

        Returns:
            Report: 完整报告
        """
        import uuid

        # Si没Tener传入 report_id，则自动Generar
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoFormat(),
        )

        # Completado的章节TítuloLista（用于进度Traza）
        completed_section_titles = []

        try:
            # Inicializando：Crear报告Archivo夹并Guardar初始Estado
            ReportManager._ensure_report_folder(report_id)

            # Inicializar el logger（结构化Log agent_log.jsonl）
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement,
            )

            # InicializandoConsolaLog记录器（console_log.txt）
            self.console_logger = ReportConsoleLogger(report_id)

            ReportManager.update_progress(
                report_id, "pending", 0, t("progress.initReport"), completed_sections=[]
            )
            ReportManager.save_report(report)

            # Fase1: 规划大纲
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id,
                "Planning",
                5,
                t("progress.startPlanningOutline"),
                completed_sections=[],
            )

            # Registrar规划InicioLog
            self.report_logger.log_Planning_start()

            if progress_callback:
                progress_callback("Planning", 0, t("progress.startPlanningOutline"))

            outline = self.Plan_outline(
                progress_callback=lambda stage, prog, msg: (
                    progress_callback(stage, prog // 5, msg)
                    if progress_callback
                    else None
                )
            )
            report.outline = outline

            # Registrar规划CompletadoLog
            self.report_logger.log_Planning_complete(outline.to_dict())

            # Guardar大纲HastaArchivo
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id,
                "Planning",
                15,
                t("progress.outlineDone", count=len(outline.sections)),
                completed_sections=[],
            )
            ReportManager.save_report(report)

            logger.info(t("report.outlineSavedToFile", reportId=report_id))

            # Fase2: 逐章节Generar（分章节Guardar）
            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []  # GuardarContenido用于上下文

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)

                # Actualizar进度
                ReportManager.update_progress(
                    report_id,
                    "geneRating",
                    base_progress,
                    t(
                        "progress.geneRatingSection",
                        title=section.title,
                        current=section_num,
                        total=total_sections,
                    ),
                    current_section=section.title,
                    completed_sections=completed_section_titles,
                )

                if progress_callback:
                    progress_callback(
                        "geneRating",
                        base_progress,
                        t(
                            "progress.geneRatingSection",
                            title=section.title,
                            current=section_num,
                            total=total_sections,
                        ),
                    )

                # Generación主章节Contenido
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg: (
                        progress_callback(
                            stage, base_progress + int(prog * 0.7 / total_sections), msg
                        )
                        if progress_callback
                        else None
                    ),
                    section_index=section_num,
                )

                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Guardar章节
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Registrar章节CompletadoLog
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip(),
                    )

                logger.info(
                    t(
                        "report.sectionSaved",
                        reportId=report_id,
                        sectionNum=f"{section_num:02d}",
                    )
                )

                # Actualizar进度
                ReportManager.update_progress(
                    report_id,
                    "geneRating",
                    base_progress + int(70 / total_sections),
                    t("progress.sectionDone", title=section.title),
                    current_section=None,
                    completed_sections=completed_section_titles,
                )

            # Fase3: 组装完整报告
            if progress_callback:
                progress_callback("geneRating", 95, t("progress.assemblingReport"))

            ReportManager.update_progress(
                report_id,
                "geneRating",
                95,
                t("progress.assemblingReport"),
                completed_sections=completed_section_titles,
            )

            # 使用ReportManager组装完整报告
            report.markdown_content = ReportManager.assemble_full_report(
                report_id, outline
            )
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoFormat()

            # 计算总耗时
            total_time_seconds = (datetime.now() - start_time).total_seconds()

            # Registrar报告CompletadoLog
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections, total_time_seconds=total_time_seconds
                )

            # GuardarMás终报告
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id,
                "completed",
                100,
                t("progress.reportComplete"),
                completed_sections=completed_section_titles,
            )

            if progress_callback:
                progress_callback("completed", 100, t("progress.reportComplete"))

            logger.info(t("report.reportGenDone", reportId=report_id))

            # CerrarConsolaLog记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(t("report.reportGenFailed", error=str(e)))
            report.status = ReportStatus.FAILED
            report.error = str(e)

            # RegistrarErrorLog
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")

            # GuardarFallidoEstado
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id,
                    "failed",
                    -1,
                    t("progress.reportFailed", error=str(e)),
                    completed_sections=completed_section_titles,
                )
            except Exception:
                pass  # IgnorarGuardarFallido的Error

            # CerrarConsolaLog记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self, message: str, chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Conversar con Report Agent

        EnPara话中AgentPuede自主调用检索工具来回答Problema

        Args:
            message: 用户Mensaje
            chat_history: Para话Historial

        Returns:
            {
                "response": "AgentResponder",
                "tool_calls": [调用的工具Lista],
                "sources": [Información来源]
            }
        """
        logger.info(t("report.agentChat", message=message[:50]))

        chat_history = chat_history or []

        # Obtener已Generar的报告Contenido
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Límite报告长度，避免上下文过长
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [报告Contenido已截断] ..."
        except Exception as e:
            logger.warning(t("report.FetchReportFailed", error=e))

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.Format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "（暂Ninguno报告）",
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # 构建Mensaje
        messages = [{"role": "system", "content": system_prompt}]

        # AgregarHistorialPara话
        for h in chat_history[-10:]:  # LímiteHistorial长度
            messages.append(h)

        # Agregar用户Mensaje
        messages.append({"role": "user", "content": message})

        # ReACTCiclo（简化版）
        tool_calls_made = []
        max_iteRations = 2  # 减少Iteraciónrondas数

        for iteRation in range(max_iteRations):
            response = self.llm.chat(messages=messages, temperature=0.5)

            # Analizar工具调用
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # 没Tener工具调用，直接VolverRespuesta
                clean_response = re.sub(
                    r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL
                )
                clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [
                        tc.get("parameters", {}).get("query", "")
                        for tc in tool_calls_made
                    ],
                }

            # Ejecutar工具调用（LímiteCantidad）
            tool_results = []
            for call in tool_calls[:1]:  # 每rondasMás多执Fila1次工具调用
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append(
                    {
                        "tool": call["name"],
                        "result": result[:1500],  # Límite结果长度
                    }
                )
                tool_calls_made.append(call)

            # 将结果添加HastaMensaje
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join(
                [f"[{r['tool']}结果]\n{r['result']}" for r in tool_results]
            )
            messages.append(
                {"role": "user", "content": observation + CHAT_OBSERVATION_SUFFIX}
            )

        # 达HastaMás大Iteración，ObtenerMás终Respuesta
        final_response = self.llm.chat(messages=messages, temperature=0.5)

        # 清理Respuesta
        clean_response = re.sub(
            r"<tool_call>.*?</tool_call>", "", final_response, flags=re.DOTALL
        )
        clean_response = re.sub(r"\[TOOL_CALL\].*?\)", "", clean_response)

        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [
                tc.get("parameters", {}).get("query", "") for tc in tool_calls_made
            ],
        }


class ReportManager:
    """
    报告管理器

    负责报告的持久化存储Y检索

    Archivo结构（分章节输出）：
    reports/
      {report_id}/
        meta.json          - 报告元InformaciónYEstado
        outline.json       - 报告大纲
        progress.json      - Generar进度
        section_01.md      - 第1章节
        section_02.md      - 第2章节
        ...
        full_report.md     - 完整报告
    """

    # Reporte存储Directorio
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, "reports")

    @classmethod
    def _ensure_reports_dir(cls):
        """确保报告根Directorio存En"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Obtener报告Archivo夹路径"""
        return os.path.join(cls.REPORTS_DIR, report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """确保报告Archivo夹存En并Volver路径"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Obtener报告元InformaciónArchivo路径"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Obtener完整报告MarkdownArchivo路径"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Obtener大纲Archivo路径"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Obtener进度Archivo路径"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Obtener章节MarkdownArchivo路径"""
        return os.path.join(
            cls._get_report_folder(report_id), f"section_{section_index:02d}.md"
        )

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Obtener Agent LogArchivo路径"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """ObtenerConsolaLogArchivo路径"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        ObtenerConsolaLogContenido

        这EsGeneración de informes过程中的Consola输出Log（INFO、WARNING等），
        Con agent_log.jsonl 的结构化Log不同。

        Args:
            report_id: 报告ID
            from_line: Desde第几FilaInicioLeer（用于增量Obtener，0 表示Desde头Inicio）

        Returns:
            {
                "logs": [LogFilaLista],
                "total_lines": 总Fila数,
                "from_line": 起始Fila号,
                "has_more": SiTodavíaTenerMás多Log
            }
        """
        log_path = cls._get_console_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # 保留原始LogFila，去掉末尾换Fila符
                    logs.append(line.rstrip("\n\r"))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # 已LeerHasta末尾
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Obtener完整的ConsolaLog（一次性Obtener全部）

        Args:
            report_id: 报告ID

        Returns:
            LogFilaLista
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obtener Agent LogContenido

        Args:
            report_id: 报告ID
            from_line: Desde第几FilaInicioLeer（用于增量Obtener，0 表示Desde头Inicio）

        Returns:
            {
                "logs": [Log条目Lista],
                "total_lines": 总Fila数,
                "from_line": 起始Fila号,
                "has_more": SiTodavíaTenerMás多Log
            }
        """
        log_path = cls._get_agent_log_path(report_id)

        if not os.path.exists(log_path):
            return {"logs": [], "total_lines": 0, "from_line": 0, "has_more": False}

        logs = []
        total_lines = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # SaltarAnalizarFallido的Fila
                        continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # 已LeerHasta末尾
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obtener完整的 Agent Log（用于一次性Obtener全部）

        Args:
            report_id: 报告ID

        Returns:
            Log条目Lista
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Guardar报告大纲

        En规划FaseCompletado后立即调用
        """
        cls._ensure_report_folder(report_id)

        with open(cls._get_outline_path(report_id), "w", encoding="utf-8") as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(t("report.outlineSaved", reportId=report_id))

    @classmethod
    def save_section(
        cls, report_id: str, section_index: int, section: ReportSection
    ) -> str:
        """
        Guardar单Elementos章节

        En每Elementos章节GenerarCompletado后立即调用，Implementación分章节输出

        Args:
            report_id: 报告ID
            section_index: 章节Índice（Desde1Inicio）
            section: 章节Objeto

        Returns:
            Guardar的Archivo路径
        """
        cls._ensure_report_folder(report_id)

        # 构建章节MarkdownContenido - 清理Posible存En的重复Título
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # GuardarArchivo
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(
            t("report.sectionFileSaved", reportId=report_id, fileSuffix=file_suffix)
        )
        return file_path

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        清理章节Contenido

        1. 移除Contenido开头Con章节Título重复的MarkdownTítuloFila
        2. 将Todos ### 及以下级别的TítuloConvertir为粗体文本

        Args:
            content: 原始Contenido
            section_title: 章节Título

        Returns:
            清理后的Contenido
        """
        import re

        if not content:
            return content

        content = content.strip()
        lines = content.split("\n")
        cleaned_lines = []
        skip_next_empty = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # InspecciónSiEsMarkdownTítuloFila
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()

                # InspecciónSiEsCon章节Título重复的Título（Saltar前5Fila内的重复）
                if i < 5:
                    if title_text == section_title or title_text.replace(
                        " ", ""
                    ) == section_title.replace(" ", ""):
                        skip_next_empty = True
                        continue

                # 将Todos级别的Título（#, ##, ###, ####等）Convertir为粗体
                # Porque章节Título由Sistema添加，Contenido中不应TenerCualquierTítulo
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Agregar空Fila
                continue

            # Si上一FilaEs被Saltar的Título，且Cuando前Fila为空，也Saltar
            if skip_next_empty and stripped == "":
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)

        # Eliminar开头的空Fila
        while cleaned_lines and cleaned_lines[0].strip() == "":
            cleaned_lines.pop(0)

        # Eliminar开头的分隔线
        while cleaned_lines and cleaned_lines[0].strip() in ["---", "***", "___"]:
            cleaned_lines.pop(0)
            # 同时移除分隔线后的空Fila
            while cleaned_lines and cleaned_lines[0].strip() == "":
                cleaned_lines.pop(0)

        return "\n".join(cleaned_lines)

    @classmethod
    def update_progress(
        cls,
        report_id: str,
        status: str,
        progress: int,
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None,
    ) -> None:
        """
        ActualizaciónGeneración de informes进度

        前端PuedeA través deLeerprogress.jsonObtener实时进度
        """
        cls._ensure_report_folder(report_id)

        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoFormat(),
        }

        with open(cls._get_progress_path(report_id), "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """ObtenerGeneración de informes进度"""
        path = cls._get_progress_path(report_id)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obtener已Generar的章节Lista

        VolverTodos已Guardar的章节ArchivoInformación
        """
        folder = cls._get_report_folder(report_id)

        if not os.path.exists(folder):
            return []

        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith("section_") and filename.endswith(".md"):
                file_path = os.path.join(folder, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # DesdeArchivo名Analizar章节Índice
                parts = filename.replace(".md", "").split("_")
                section_index = int(parts[1])

                sections.append(
                    {
                        "filename": filename,
                        "section_index": section_index,
                        "content": content,
                    }
                )

        return sections

    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        组装完整报告

        Desde已Guardar的章节Archivo组装完整报告，并进FilaTítulo清理
        """
        folder = cls._get_report_folder(report_id)

        # 构建报告头部
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"

        # 按SecuencialLeerTodos章节Archivo
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]

        # 后Procesar：清理整Elementos报告的TítuloProblema
        md_content = cls._post_process_report(md_content, outline)

        # Guardar完整报告
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(t("report.fullReportAssembled", reportId=report_id))
        return md_content

    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        后Procesar报告Contenido

        1. 移除重复的Título
        2. 保留报告主Título(#)Y章节Título(##)，移除Otro级别的Título(###, ####等)
        3. 清理多余的空FilaY分隔线

        Args:
            content: 原始报告Contenido
            outline: 报告大纲

        Returns:
            Procesar后的Contenido
        """
        import re

        lines = content.split("\n")
        processed_lines = []
        prev_was_heading = False

        # 收集大纲中的Todos章节Título
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # InspecciónSiEsTítuloFila
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # InspecciónSiEs重复Título（En连续5Fila内出现相同Contenido的Título）
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r"^(#{1,6})\s+(.+)$", prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break

                if is_duplicate:
                    # Saltar重复Título及其后的空Fila
                    i += 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    continue

                # Título层级Procesar：
                # - # (level=1) Solo保留报告主Título
                # - ## (level=2) 保留章节Título
                # - ### 及以下 (level>=3) Convertir为粗体文本

                if level == 1:
                    if title == outline.title:
                        # 保留报告主Título
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # 章节TítuloError使用了#，修正为##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Otro一级Título转为粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # 保留章节Título
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # 非章节的二级Título转为粗体
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### 及以下级别的TítuloConvertir为粗体文本
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == "---" and prev_was_heading:
                # SaltarTítulo后紧跟的分隔线
                i += 1
                continue

            elif stripped == "" and prev_was_heading:
                # Título后Solo保留一Elementos空Fila
                if processed_lines and processed_lines[-1].strip() != "":
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1

        # 清理连续的多Elementos空Fila（保留Más多2Elementos）
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == "":
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)

        return "\n".join(result_lines)

    @classmethod
    def save_report(cls, report: Report) -> None:
        """Guardar报告元InformaciónY完整报告"""
        cls._ensure_report_folder(report.report_id)

        # Guardar元InformaciónJSON
        with open(cls._get_report_path(report.report_id), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Guardar大纲
        if report.outline:
            cls.save_outline(report.report_id, report.outline)

        # Guardar完整Markdown报告
        if report.markdown_content:
            with open(
                cls._get_report_markdown_path(report.report_id), "w", encoding="utf-8"
            ) as f:
                f.write(report.markdown_content)

        logger.info(t("report.reportSaved", reportId=report.report_id))

    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Obtener报告"""
        path = cls._get_report_path(report_id)

        if not os.path.exists(path):
            # 兼容旧格式：Verificar直接存储EnreportsDirectorio下的Archivo
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 重建ReportObjeto
        outline = None
        if data.get("outline"):
            outline_data = data["outline"]
            sections = []
            for s in outline_data.get("sections", []):
                sections.append(
                    ReportSection(title=s["title"], content=s.get("content", ""))
                )
            outline = ReportOutline(
                title=outline_data["title"],
                summary=outline_data["summary"],
                sections=sections,
            )

        # Simarkdown_content为空，尝试Desdefull_report.mdLeer
        markdown_content = data.get("markdown_content", "")
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()

        return Report(
            report_id=data["report_id"],
            simulation_id=data["simulation_id"],
            graph_id=data["graph_id"],
            simulation_requirement=data["simulation_requirement"],
            status=ReportStatus(data["status"]),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
            error=data.get("error"),
        )

    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Basado enSimulaciónIDObtener报告"""
        cls._ensure_reports_dir()

        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新格式：Archivo夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # 兼容旧格式：JSONArchivo
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report

        return None

    @classmethod
    def list_reports(
        cls, simulation_id: Optional[str] = None, limit: int = 50
    ) -> List[Report]:
        """Columna出报告"""
        cls._ensure_reports_dir()

        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # 新格式：Archivo夹
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # 兼容旧格式：JSONArchivo
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)

        # 按CrearTiempo倒序
        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Eliminación报告（整archivos夹）"""
        import shutil

        folder_path = cls._get_report_folder(report_id)

        # 新格式：Eliminación整archivos夹
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(t("report.reportFolderDeleted", reportId=report_id))
            return True

        # 兼容旧格式：Eliminación单独的Archivo
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")

        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True

        return deleted
