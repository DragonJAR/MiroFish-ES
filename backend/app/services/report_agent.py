"""
Report Agent Service
Generación de informes de simulación usando LangChain + Zep con patrón ReACT

Funcionalidades:
1. Genera informes basados en requisitos de simulación e información del grafo Zep
2. Primero planifica la estructura del índice, luego genera por secciones
3. Cada sección usa el patrón ReACT de pensamiento y reflexión multicapa
4. Soporta diálogo con el usuario, invocando herramientas de búsqueda de forma autónoma
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
from .zep_tools import (
    ZepToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult,
)

# Intentar importar el sistema de prompts i18n
try:
    from ..prompts import load_prompt as _load_prompt
    from ..prompts import get_prompt as _get_prompt

    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False
    _load_prompt = None
    _get_prompt = None

logger = get_logger("mirofish.report_agent")


class ReportLogger:
    """
    Report Agent Detailed Logger

    Genera archivos agent_log.jsonl en la carpeta del informe, registrando cada acción detallada.
    Cada línea es un objeto JSON completo con marca de tiempo, tipo de acción, detalles, etc.
    """

    def __init__(self, report_id: str):
        """
        Inicializar logger

        Args:
            report_id: ID del informe, usado para determinar la ruta del archivo de log
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
            action: Tipo de acción como 'start', 'tool_call', 'llm_response', 'section_complete'
            stage: Etapa actual como 'planning', 'generating', 'completed'
            details: Diccionario de detalles, sin truncar
            section_title: Título de la sección actual (opcional)
            section_index: Índice de la sección actual (opcional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details,
        }

        # Escribir append en archivo JSONL
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Registrar inicio de generación del informe"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Tarea de generación del informe iniciada",
            },
        )

    def log_planning_start(self):
        """Registrar inicio de planificación del índice"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Iniciando planificación del índice del informe"},
        )

    def log_planning_context(self, context: Dict[str, Any]):
        """Registrar información de contexto obtenida durante la planificación"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Obteniendo información de contexto de simulación",
                "context": context,
            },
        )

    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Registrar completación de planificación del índice"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Planificación del índice completada",
                "outline": outline_dict,
            },
        )

    def log_section_start(self, section_title: str, section_index: int):
        """Registrar inicio de generación de sección"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Iniciando generación de sección: {section_title}"},
        )

    def log_react_thought(
        self, section_title: str, section_index: int, iteration: int, thought: str
    ):
        """Registrar proceso de pensamiento ReACT"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT Ronda {iteration} de pensamiento",
            },
        )

    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int,
    ):
        """Registrar invocación de herramienta"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Invocando herramienta: {tool_name}",
            },
        )

    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int,
    ):
        """Registrar resultado de invocación de herramienta (contenido completo, sin truncar)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Resultado completo, sin truncar
                "result_length": len(result),
                "message": f"Herramienta {tool_name} retornó resultado",
            },
        )

    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool,
    ):
        """Registrar respuesta LLM (contenido completo, sin truncar)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Respuesta completa, sin truncar
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"Respuesta LLM (invocaciones: {has_tool_calls}, respuesta final: {has_final_answer})",
            },
        )

    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int,
    ):
        """Registrar generación de contenido de sección completada (solo registra contenido, no toda la sección)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Contenido completo, sin truncar
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Contenido de sección {section_title} generado",
            },
        )

    def log_section_full_complete(
        self, section_title: str, section_index: int, full_content: str
    ):
        """
        Registrar completación de generación de sección

        El frontend debe escuchar este log para determinar si una sección está realmente completa y obtener el contenido completo
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Sección {section_title} generada completamente",
            },
        )

    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Registrar completación de generación del informe"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Generación del informe completada",
            },
        )

    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Registrar error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error ocurrido: {error_message}",
            },
        )


class ReportConsoleLogger:
    """
    Report Agent Console Logger

    Escribe logs de estilo consola (INFO, WARNING, etc.) en el archivo console_log.txt en la carpeta del informe.
    Estos logs son diferentes a agent_log.jsonl, son salida de consola en formato texto plano.
    """

    def __init__(self, report_id: str):
        """
        Inicializar logger de consola

        Args:
            report_id: ID del informe, usado para determinar la ruta del archivo de log
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
        """Configurar handler de archivo para escribir logs simultáneamente"""
        import logging

        # Crear file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path, mode="a", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.INFO)

        # Usar formato idéntico al de consola
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        self._file_handler.setFormatter(formatter)

        # Agregar a loggers relacionados con report_agent
        loggers_to_attach = [
            "mirofish.report_agent",
            "mirofish.zep_tools",
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Evitar duplicación
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)

    def close():
        """Cerrar file handler y remover del logger"""
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
        """Asegurar cierre del file handler al destruir"""
        self.close()


class ReportStatus(str, Enum):
    """Estado del informe"""

    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Sección del informe"""

    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "content": self.content}

    def to_markdown(self, level: int = 2) -> str:
        """Convertir a formato Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Índice del informe"""

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
        """Convertir a formato Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Informe completo"""

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
# Prompt Templates Constants
# ═══════════════════════════════════════════════════════════════

# ── Tool Descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
【Busqueda en Profundidad - Potente herramienta de busqueda】
Esta es nuestra poderosa funcion de busqueda, disenada para analisis profundo. Ella:
1. Dividira automaticamente tu pregunta en multiples sub-preguntas
2. Buscara informacion en multiples dimensiones en el grafo de simulacion
3. Integrara resultados de busqueda semantica, analisis de entidades y seguimiento de cadenas de relaciones
4. Retornara el contenido de busqueda mas completo y profundo

【Casos de uso】
- Necesitas analizar profundamente un tema
- Necesitas entender multiples aspectos de un evento
- Necesitas obtener material rico para sustentar secciones del informe

【Contenido retornado】
- Hechos originales relacionados (citables directamente)
- Entidades clave con insights
- Analisis de cadenas de relaciones"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Busqueda de Amplitud - Obtener vista completa】
Esta herramienta se usa para obtener la vision completa de los resultados de simulacion, especialmente adecuada para entender la evolucion de eventos. Ella:
1. Obtiene todos los nodos y relaciones relacionados
2. Distingue entre hechos actualmente validos e historicos/expirados
3. Te ayuda a entender como evoluciona la opinion publica

【Casos de uso】
- Necesitas entender el desarrollo completo de un evento
- Necesitas comparar cambios de opinion en diferentes etapas
- Necesitas obtener informacion completa de entidades y relaciones

【Contenido retornado】
- Hechos actualmente validos (resultados mas recientes de simulacion)
- Hechos historicos/expirados (registros de evolucion)
- Todas las entidades involucradas"""

TOOL_DESC_QUICK_SEARCH = """\
【Busqueda Simple - Busqueda rapida】
Herramienta de busqueda ligera y rapida, adecuada para consultas de informacion simples y directas.

【Casos de uso】
- Necesitas buscar rapidamente una informacion especifica
- Necesitas verificar un hecho
- Busqueda simple de informacion

【Contenido retornado】
- Lista de hechos mas relevantes a la consulta"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【Entrevista Profunda - Entrevista real de Agent (Doble plataforma)】
Invoca la API de entrevista del entorno de simulacion OASIS para realizar entrevistas reales a los Agents en ejecucion!
No es simulacion LLM, sino invocar la interfaz de entrevista real para obtener respuestas originales de los Agents simulados.
Por defecto entrevista simultaneamente en Twitter y Reddit para obtener perspectivas mas completas.

Flujo de funciones:
1. Lee automaticamente archivos de perfil para conocer todos los Agents simulados
2. Selecciona inteligentemente los Agents mas relevantes al tema de entrevista (estudiantes, medios, oficiales, etc.)
3. Genera automaticamente preguntas de entrevista
4. Invoca la interfaz /api/simulation/interview/batch para entrevista real en doble plataforma
5. Integra todos los resultados de entrevista, proporcionando analisis multi-perspectiva

【Casos de uso】
- Necesitas entender la opinion del evento desde diferentes perspectivas de roles (que piensan los estudiantes? que dicen los medios? que dicen los oficiales?)
- Necesitas recolectar opiniones y posiciones de multiples partes
- Necesitas obtener respuestas reales de Agents simulados (del entorno de simulacion OASIS)
- Quieres que el informe sea mas vivo, incluyendo "registros de entrevista"

【Contenido retornado】
- Informacion de identidad del Agent entrevistado
- Respuestas de entrevista del Agent en las dos plataformas Twitter y Reddit
- Citas clave (citables directamente)
- Resumen de entrevista y comparacion de perspectivas

【Importante】Se necesita que el entorno de simulacion OASIS este en ejecucion para usar esta funcion!"""

# ── Plantillas de prompts de planificación del índice ──

PLAN_SYSTEM_PROMPT = """\
Eres un experto en redaccion de informes de prediccion futura, con una perspectiva de "vista divina" sobre el mundo simulado: puedes observar el comportamiento, palabras e interacciones de cada Agente en la simulacion.

【Concepto Central】
Hemos construido un mundo simulado e inyectado requisitos de simulacion especificos como variables. La evolucion del mundo simulado es la prediccion de lo que podria suceder en el futuro. No estas observando "datos experimentales", sino un "ensayo del futuro".

【Tu Tarea】
Redactar un "Informe de Prediccion Futura", respondiendo:
1. Bajo nuestras condiciones establecidas, que sucede en el futuro?
2. Como reaccionan y actuan los diferentes tipos de Agentes (grupos)?
3. Que tendencias futuras riesgosas se revelan?

【Posicionamiento del Informe】
- ✅ Este es un informe de prediccion futura basado en simulacion, revelando "si esto sucede, como sera el futuro"
- ✅ Se centra en resultados de prediccion: direccion de eventos, reacciones grupales, fenomenos emergentes, riesgos potenciales
- ✅ Las palabras y acciones de los Agentes en el mundo simulado son predicciones del comportamiento futuro de las personas
- ❌ No es un analisis del estado actual del mundo real
- ❌ No es un resumen general de opinion publica

【Limite de Secciones】
- Minimo 2 secciones, maximo 5 secciones
- No se necesitan subsecciones, cada seccion redacta contenido completo directamente
- El contenido debe ser conciso, centrado en hallazgos clave de la prediccion
- La estructura de secciones la diseñas tu segun los resultados de la prediccion

Por favor genera el esquema del informe en formato JSON:
{
    "title": "Titulo del informe",
    "summary": "Resumen del informe (una frase que resuma los hallazgos clave de la prediccion)",
    "sections": [
        {
            "title": "Titulo de la seccion",
            "description": "Descripcion del contenido de la seccion"
        }
    ]
}

Nota: El array de sections debe tener entre 2 y 5 elementos!"""

PLAN_USER_PROMPT_TEMPLATE = """\
【Configuracion del Escenario de Prediccion】
La variable que inyectamos al mundo simulado (requisitos de simulacion): {simulation_requirement}

【Escala del Mundo Simulado】
- Numero de entidades participando en la simulacion: {total_nodes}
- Numero de relaciones generadas entre entidades: {total_edges}
- Distribucion de tipos de entidades: {entity_types}
- Numero de Agentes activos: {total_entities}

【Muestra de algunos hechos futuros predichos por la simulacion】
{related_facts_json}

Por favor examina este ensayo del futuro desde la "perspectiva de vista divina":
1. Bajo nuestras condiciones establecidas, que estado presenta el futuro?
2. Como reaccionan y actuan los diferentes grupos de personas (Agentes)?
3. Que tendencias futuras riesgosas se revelan?

Disena la estructura de secciones del informe mas adecuada segun los resultados de la prediccion.

【Recordatorio】Cantidad de secciones del informe: minimo 2, maximo 5, el contenido debe ser conciso y centrado en hallazgos clave de la prediccion."""

# ── Plantillas de prompts de generación de sección ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Eres un experto en redaccion de informes de prediccion futura, redactando una seccion del informe.

Titulo del informe: {report_title}
Resumen del informe: {report_summary}
Escenario de prediccion (requisitos de simulacion): {simulation_requirement}

Seccion actual a redactar: {section_title}

============================================================
【Concepto Central】
============================================================

El mundo simulado es un ensayo del futuro. Hemos inyectado condiciones especificas (requisitos de simulacion) al mundo simulado,
el comportamiento e interacciones de los Agentes en la simulacion son predicciones del comportamiento futuro de las personas.

Tu tarea es:
- Revelar que sucede en el futuro bajo las condiciones establecidas
- Predecir como reaccionan y actuan los diferentes grupos de personas (Agentes)
- Descubrir tendencias futuras riesgosas y oportunidades

❌ No escribir como un analisis del estado actual del mundo real
✅ Centrate en "como sera el futuro" — los resultados de la simulacion son el futuro predicho

============================================================
【Regla Mas Importante - Debe Cumplirse】
============================================================

1. 【Debes llamar herramientas para observar el mundo simulado】
   - Estas observando el ensayo del futuro desde la "perspectiva de vista divina"
   - Todo el contenido debe provenir de eventos y palabras/acciones de Agentes que ocurrieron en el mundo simulado
   - Esta prohibido usar tu propio conocimiento para redactar el contenido del informe
   - Cada seccion debe llamar al menos 3 veces herramientas (maximo 5) para observar el mundo simulado, que representa el futuro

2. 【Debes citar las palabras y acciones originales de los Agentes】
   - Las palabras y acciones de los Agentes son predicciones del comportamiento futuro de las personas
   - Usa formato de cita en el informe para mostrar estas predicciones, por ejemplo:
     > "Un tipo de grupo dira: contenido original..."
   - Estas citas son la evidencia central de la prediccion de la simulacion

3. 【Consistencia del idioma - El contenido citado debe traducirse al idioma del informe】
   - El contenido devuelto por las herramientas puede contener expresiones en ingles o mezclado chino-ingles
   - Si los requisitos de simulacion y el material original son en chino, el informe debe escribirse completamente en chino
   - Cuando cites contenido en ingles o mezclado chino-ingles devuelto por herramientas, debes traducirlo a chino fluido antes de escribirlo en el informe
   - Al traducir, mantén el significado original, asegurando que la expresion sea natural y fluida
   - Esta regla se aplica tanto al texto principal como al contenido en bloques de cita (formato >)

4. 【Presentar fielmente los resultados de la prediccion】
   - El contenido del informe debe reflejar los resultados de la simulacion que representan el futuro en el mundo simulado
   - No agregar informacion que no exista en la simulacion
   - Si falta informacion en algunos aspectos, indicarlo honestamente

============================================================
【⚠️ Especificaciones de Formato - Extremadamente Importante!】
============================================================

【Una seccion = Unidad minima de contenido】
- Cada seccion es la unidad minima de division del informe
- ❌ Prohibido usar cualquier titulo de Markdown dentro de las secciones (#, ##, ###, ####, etc.)
- ❌ Prohibido agregar el titulo principal de la seccion al inicio del contenido
- ✅ Los titulos de seccion son agregados automaticamente por el sistema, solo necesitas redactar contenido de texto plano
- ✅ Usa **negritas**, separacion de parrafos, citas, listas para organizar el contenido, pero no titulos

【Ejemplo Correcto】
Este capitulo analiza la situacion de difusion de opinion publica del evento. A traves del analisis profundo de los datos de simulacion, descubrimos...

**Fase de inicio explosivo**

Weibo, como primer escenario de opinion publica, asumio la funcion principal de volumen inicial de informacion:

> "Weibo contribute with 68% of volumen inicial..."

**Fase de amplificacion emocional**

La plataforma Douyin amplifico mas la influencia del evento:

- Fuerte impacto visual
- Alta resonancia emocional

【Ejemplo Incorrecto】
## Resumen Ejecutivo          ← Incorrecto! No agregar ningun titulo
### Fase de Inicio     ← Incorrecto! No usar ### para subsecciones
#### 1.1 Analisis Detallado   ← Incorrecto! No usar #### para subdivisions

Este capitulo analiza...

============================================================
【Herramientas de Recuperacion Disponibles】(llamar 3-5 veces por seccion)
============================================================

{tools_description}

【Sugerencias de uso de herramientas - Por favor usa diferentes herramientas mezcladas, no solo una】
- insight_forge: Analisis de profundidad, descompone automaticamente el problema y recupera hechos y relaciones en multiples dimensiones
- panorama_search: Busqueda panoramica de amplitud, comprende la vision completa, linea de tiempo y proceso de evolucion del evento
- quick_search: Verificacion rapida de un punto especifico de informacion
- interview_agents: Entrevistar Agentes simulados, obtener perspectivas de primera persona de diferentes roles y reacciones reales

============================================================
【Flujo de Trabajo】
============================================================

Cada vez solo puedes hacer UNA de las siguientes dos cosas (no ambas al mismo tiempo):

Opcion A - Llamar herramienta:
Escribe tu pensamiento, luego usa el siguiente formato para llamar una herramienta:
<tool_call>
{{"name": "nombre_herramienta", "parameters": {{"parametro": "valor"}}}}
</tool_call>
El sistema ejecutara la herramienta y te devolvera el resultado. No necesitas ni debes escribir tu mismo el resultado devuelto por la herramienta.

Opcion B - Salida de contenido final:
Cuando hayas obtenido suficiente informacion a traves de herramientas, comienza con "Final Answer:" para salida del contenido de la seccion.

⚠️ Estrictamente prohibido:
- Prohibido incluir llamada de herramienta y Final Answer en una misma respuesta
- Prohibido inventarte el resultado de la herramienta (Observation), todos los resultados de herramientas son inyectados por el sistema
- Cada respuesta puede llamar maximo una herramienta

============================================================
【Requisitos del Contenido de la Seccion】
============================================================

1. El contenido debe basarse en datos de simulacion recuperados por herramientas
2. Citar extensamente el texto original para mostrar los efectos de la simulacion
3. Usar formato Markdown (pero prohibido usar titulos):
   - Usar **texto en negritas** para marcar puntos clave (en lugar de subtitulos)
   - Usar listas (- o 1.2.3.) para organizar puntos
   - Usar lineas vacias para separar diferentes parrafos
   - ❌ Prohibido usar #, ##, ###, #### o cualquier sintaxis de titulos
4. 【Especificaciones de formato de citas - Deben ser parrafos independientes】
   Las citas deben ser parrafos independientes, con una linea vacia antes y despues, no mezcladas en parrafos:

   ✅ Formato correcto:
   La respuesta de la universidad fue considerada carente de contenido sustancial.

   > "El patron de respuesta de la universidad parecio rigido y lento en el entorno cambiante de redes sociales."

   Esta evaluacion refleja la insatisfaccion general del publica.

   ❌ Formato incorrecto:
   La respuesta de la universidad fue considerada carente de contenido sustancial.> "El patron de respuesta..." Esta evaluacion refleja...

5. Mantener coherencia logica con otras secciones
6. 【Evitar repeticion】Lee cuidadosamente el contenido de las secciones completadas a continuacion, no describir la misma informacion
7. 【Reiteracion】No agregar ningun titulo! Usar **negritas** en lugar de titulos de subsecciones"""

SECTION_USER_PROMPT_TEMPLATE = """\
Contenido de secciones completadas (por favor lee cuidadosamente para evitar repeticion):
{previous_content}

============================================================
【Tarea Actual】 Redactar seccion: {section_title}
============================================================

【Recordatorio Importante】
1. Lee cuidadosamente las secciones completadas arriba, evita repetir el mismo contenido!
2. Antes de comenzar, debes primero llamar herramientas para obtener datos de simulacion
3. Por favor usa diferentes herramientas mezcladas, no solo una
4. El contenido del informe debe provenir de resultados de busqueda, no uses tu propio conocimiento

【⚠️ Advertencia de Formato - Debe Cumplirse】
- ❌ No escribir ningun titulo (#, ##, ###, #### tampoc)
- ❌ No escribir "{section_title}" como inicio
- ✅ Los titulos de seccion son agregados automaticamente por el sistema
- ✅ Escribe directamente el texto, usa **negritas** en lugar de titulos de subsecciones

Por favor comienza:
1. Primero piensa (Thought) que informacion necesita esta seccion
2. Luego llama herramientas (Action) para obtener datos de simulacion
3. Despues de recopilar suficiente informacion, salida Final Answer (texto plano, sin ningun titulo)"""

# ── Plantilla de mensajes dentro del ciclo ReACT ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (resultados de busqueda):

═══ Herramienta {tool_name} retorno ═══
{result}

══════════════════════════════════════════════════════════════
Herramientas llamadas {tool_calls_count}/{max_tool_calls} (usadas: {used_tools_str}){unused_hint}
- Si la informacion es suficiente: comienza con "Final Answer:" para salida del contenido de la seccion (debes citar el texto original de arriba)
- Si necesitas mas informacion: llama una herramienta para continuar buscando
══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Nota】Solo has llamado {tool_calls_count} herramientas, necesitas al menos {min_tool_calls}."
    "Por favor llama mas herramientas para obtener mas datos de simulacion, luego salida Final Answer. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Actualmente solo se han llamado {tool_calls_count} herramientas, necesitas al menos {min_tool_calls}."
    "Por favor llama herramientas para obtener datos de simulacion. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Se ha alcanzado el limite de llamadas de herramientas ({tool_calls_count}/{max_tool_calls}), no se pueden llamar mas herramientas."
    'Por favor basandote en la informacion ya obtenida, comienza con "Final Answer:" para salida del contenido de la seccion.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Aun no has usado: {unused_list}, se recomienda probar diferentes herramientas para obtener informacion desde multiples angulos"

REACT_FORCE_FINAL_MSG = "Has alcanzado el limite de llamadas de herramientas, por favor salida directamente Final Answer: y genera el contenido de la seccion."

# ── Chat prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Eres un asistente de prediccion de simulacion eficiente y conciso.

【Contexto】
Condiciones de prediccion: {simulation_requirement}

【Informe de analisis ya generado】
{report_content}

【Reglas】
1. Prioriza responder preguntas basandote en el contenido del informe de arriba
2. Responde directamente, evita discourses largos de reflexion
3. Solo cuando el contenido del informe sea insuficiente para responder, llama herramientas para buscar mas datos
4. Las respuestas deben ser concisas, claras y organizadas

【Herramientas disponibles】(usar solo cuando sea necesario, llamar maximo 1-2 veces)
{tools_description}

【Formato de llamada de herramientas】
<tool_call>
{{"name": "nombre_herramienta", "parameters": {{"parametro": "valor"}}}}
</tool_call>

【Estilo de respuesta】
- Conciso y directo, sin discursos largos
- Usar formato > para citar contenido clave
- Prioriza dar conclusiones, luego explica razones"""

CHAT_OBSERVATION_SUFFIX = "\n\nPor favor responde la pregunta de manera concisa."


# ═══════════════════════════════════════════════════════════════
# ReportAgent Clase Principal
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Agente de Generacion de Informes de Simulacion

    Usa el patron ReACT (Reasoning + Acting):
    1. Fase de planificacion: Analiza requisitos de simulacion, planifica estructura del indice
    2. Fase de generacion: Genera contenido seccion por seccion, cada seccion puede llamar herramientas multiples veces
    3. Fase de reflexion: Verifica integridad y precision del contenido
    """

    # Maximo de llamadas a herramientas (por seccion)
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Maximo de rondas de reflexion
    MAX_REFLECTION_ROUNDS = 3

    # Maximo de llamadas a herramientas en conversacion
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
        Inicializar Report Agent

        Args:
            graph_id: ID del grafo
            simulation_id: ID de simulacion
            simulation_requirement: Descripcion de requisitos de simulacion
            llm_client: Cliente LLM (opcional)
            zep_tools: Servicio de herramientas Zep (opcional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()

        # Definicion de herramientas
        self.tools = self._define_tools()

        # Registrador de informe (se inicializa en generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Registrador de consola (se inicializa en generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(
            f"ReportAgent inicializado: graph_id={graph_id}, simulation_id={simulation_id}"
        )

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Definir herramientas disponibles"""
        # Cargar descripciones con i18n si está disponible
        if _PROMPTS_AVAILABLE:
            desc_insight = _get_prompt("report", "tool_insight_forge", "")
            desc_panorama = _get_prompt("report", "tool_panorama", "")
            desc_quick = _get_prompt("report", "tool_quick", "")
            desc_interview = _get_prompt("report", "tool_interview", "")
        else:
            desc_insight = TOOL_DESC_INSIGHT_FORGE
            desc_panorama = TOOL_DESC_PANORAMA_SEARCH
            desc_quick = TOOL_DESC_QUICK_SEARCH
            desc_interview = TOOL_DESC_INTERVIEW_AGENTS

        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": desc_insight,
                "parameters": {
                    "query": "El problema o tema que quieres analizar en profundidad",
                    "report_context": "Contexto de la seccion actual del informe (opcional, ayuda a generar sub-preguntas mas precisas)",
                },
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": desc_panorama,
                "parameters": {
                    "query": "Consulta de busqueda, para ordenamiento por relevancia",
                    "include_expired": "Si incluir contenido expirado/historico (por defecto True)",
                },
            },
            "quick_search": {
                "name": "quick_search",
                "description": desc_quick,
                "parameters": {
                    "query": "Cadena de consulta de busqueda",
                    "limit": "Cantidad de resultados a retornar (opcional, por defecto 10)",
                },
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": desc_interview,
                "parameters": {
                    "interview_topic": "Tema de entrevista o descripcion del requerimiento (ej: 'conocer la opinion de estudiantes sobre el incidente de formaldehido en dormitorios')",
                    "max_agents": "Cantidad maxima de Agents a entrevistar (opcional, por defecto 5, maximo 10)",
                },
            },
        }

    def _execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], report_context: str = ""
    ) -> str:
        """
        Ejecutar llamada a herramienta

        Args:
            tool_name: Nombre de herramienta
            parameters: Parametros de herramienta
            report_context: Contexto del informe (para InsightForge)

        Returns:
            Resultado de la herramienta (formato texto)
        """
        logger.info(f"Ejecutando herramienta: {tool_name}, parametros: {parameters}")

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
                # Busqueda de amplitud - obtener vision completa
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ["true", "1", "yes"]
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id, query=query, include_expired=include_expired
                )
                return result.to_text()

            elif tool_name == "quick_search":
                # Busqueda simple - recuperacion rapida
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id, query=query, limit=limit
                )
                return result.to_text()

            elif tool_name == "interview_agents":
                # Entrevista profunda - llama a la API real de OASIS para obtener respuestas de Agentes simulados (doble plataforma)
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

            # ========== Herramientas heredadas retrocompatibles (redirigidas internamente) ==========

            elif tool_name == "search_graph":
                # Redirigir a quick_search
                logger.info("search_graph redirigido a quick_search")
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
                # Redirigir a insight_forge, porque es mas potente
                logger.info("get_simulation_context redirigido a insight_forge")
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
                return f"Herramienta desconocida: {tool_name}. Por favor usa una de las siguientes herramientas: insight_forge, panorama_search, quick_search"

        except Exception as e:
            logger.error(
                f"Error en ejecucion de herramienta: {tool_name}, error: {str(e)}"
            )
            return f"Error en ejecucion de herramienta: {str(e)}"

    # Conjunto de nombres de herramientas validos, para validacion en parsing JSON plano
    VALID_TOOL_NAMES = {
        "insight_forge",
        "panorama_search",
        "quick_search",
        "interview_agents",
    }

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parsear llamadas a herramientas desde la respuesta LLM

        Formatos soportados (por prioridad):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. JSON plano (la respuesta completa o una sola linea es un JSON de llamada a herramienta)
        """
        tool_calls = []

        # Formato 1: Estilo XML (formato estandar)
        xml_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Formato 2: Fallback - LLM salida JSON plano (sin etiquetas <tool_call>)
        # Solo intentar si el formato 1 no coincide, evitar falso positivo en el texto
        stripped = response.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # La respuesta puede contener texto de pensamiento + JSON plano, intentar extraer el ultimo objeto JSON
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
        """Validar si el JSON parseado es una llamada a herramienta valida"""
        # Soporta {"name": ..., "parameters": ...} y {"tool": ..., "params": ...} dos formatos de claves
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Unificar claves a name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False

    def _get_tools_description(self) -> str:
        """Generar texto de descripcion de herramientas"""
        desc_parts = ["Herramientas disponibles:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join(
                [f"{k}: {v}" for k, v in tool["parameters"].items()]
            )
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parametros: {params_desc}")
        return "\n".join(desc_parts)

    def plan_outline(
        self, progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Planificar esquema del informe

        Usa LLM para analizar requisitos de simulacion, planificar estructura del indice

        Args:
            progress_callback: Funcion de callback de progreso

        Returns:
            ReportOutline: Esquema del informe
        """
        logger.info("Iniciando planificación del esquema del informe...")

        if progress_callback:
            progress_callback("planning", 0, "Analizando requisitos de simulación...")

        # Primero obtener el contexto de simulación
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id, simulation_requirement=self.simulation_requirement
        )

        if progress_callback:
            progress_callback("planning", 30, "Generando esquema del informe...")

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            system_prompt = _get_prompt("report", "plan_system", "")
            user_prompt = _load_prompt(
                "report",
                "plan_user",
                simulation_requirement=self.simulation_requirement,
                total_nodes=context.get("graph_statistics", {}).get("total_nodes", 0),
                total_edges=context.get("graph_statistics", {}).get("total_edges", 0),
                entity_types=list(
                    context.get("graph_statistics", {}).get("entity_types", {}).keys()
                ),
                total_entities=context.get("total_entities", 0),
                related_facts_json=json.dumps(
                    context.get("related_facts", [])[:10], ensure_ascii=False, indent=2
                ),
            )
        else:
            system_prompt = PLAN_SYSTEM_PROMPT
            user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
                simulation_requirement=self.simulation_requirement,
                total_nodes=context.get("graph_statistics", {}).get("total_nodes", 0),
                total_edges=context.get("graph_statistics", {}).get("total_edges", 0),
                entity_types=list(
                    context.get("graph_statistics", {}).get("entity_types", {}).keys()
                ),
                total_entities=context.get("total_entities", 0),
                related_facts_json=json.dumps(
                    context.get("related_facts", [])[:10], ensure_ascii=False, indent=2
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
                progress_callback(
                    "planning", 80, "Analizando estructura del esquema..."
                )

            # Parsear esquema
            sections = []
            for section_data in response.get("sections", []):
                sections.append(
                    ReportSection(title=section_data.get("title", ""), content="")
                )

            outline = ReportOutline(
                title=response.get("title", "Informe de Analisis de Simulacion"),
                summary=response.get("summary", ""),
                sections=sections,
            )

            if progress_callback:
                progress_callback(
                    "planning", 100, "Planificación del esquema completada"
                )

            logger.info(
                f"Planificación del esquema completada: {len(sections)} secciones"
            )
            return outline

        except Exception as e:
            logger.error(f"Error en planificacion del esquema: {str(e)}")
            # Devolver esquema por defecto (3 secciones, como fallback)
            return ReportOutline(
                title="Informe de Prediccion Futura",
                summary="Analisis de tendencias futuras y riesgos basado en simulacion",
                sections=[
                    ReportSection(title="Escenario de Prediccion y Hallazgos Clave"),
                    ReportSection(
                        title="Analisis de Prediccion de Comportamiento Grupal"
                    ),
                    ReportSection(
                        title="Perspectivas de Tendencias y Avisos de Riesgo"
                    ),
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
        Generar contenido de una seccion usando patron ReACT

        Ciclo ReACT:
        1. Thought (pensar) - Analizar que informacion se necesita
        2. Action (actuar) - Llamar herramientas para obtener informacion
        3. Observation (observar) - Analizar resultados de herramientas
        4. Repetir hasta tener suficiente informacion o alcanzar maximo de veces
        5. Final Answer (respuesta final) - Generar contenido de seccion

        Args:
            section: Seccion a generar
            outline: Esquema completo
            previous_sections: Contenido de secciones anteriores (para mantener coherencia)
            progress_callback: Callback de progreso
            section_index: Indice de seccion (para registro)

        Returns:
            Contenido de seccion (formato Markdown)
        """
        logger.info(f"ReACT generando seccion: {section.title}")

        # Registrar inicio de seccion
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            system_prompt = _load_prompt(
                "report",
                "section_system",
                report_title=outline.title,
                report_summary=outline.summary,
                simulation_requirement=self.simulation_requirement,
                section_title=section.title,
                tools_description=self._get_tools_description(),
            )
        else:
            system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
                report_title=outline.title,
                report_summary=outline.summary,
                simulation_requirement=self.simulation_requirement,
                section_title=section.title,
                tools_description=self._get_tools_description(),
            )

        # Construir prompt de usuario - maximo 4000 caracteres por cada seccion completada
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Maximo 4000 caracteres por seccion
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(Esta es la primera seccion)"

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            user_prompt = _load_prompt(
                "report",
                "section_user",
                previous_content=previous_content,
                section_title=section.title,
            )
        else:
            user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
                previous_content=previous_content,
                section_title=section.title,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Ciclo ReACT
        tool_calls_count = 0
        max_iterations = 5  # Maximo de iteraciones
        min_tool_calls = 3  # Minimo de llamadas a herramientas
        conflict_retries = 0  # Conflictos consecutivos de llamada a herramienta y Final Answer simultaneos
        used_tools = set()  # Registrar nombres de herramientas ya llamadas
        all_tools = {
            "insight_forge",
            "panorama_search",
            "quick_search",
            "interview_agents",
        }

        # Contexto del informe, para generacion de subpreguntas de InsightForge
        report_context = f"Titulo de seccion: {section.title}\nRequisitos de simulacion: {self.simulation_requirement}"

        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating",
                    int((iteration / max_iterations) * 100),
                    f"Busqueda profunda y redaccion en progreso ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})",
                )

            # Llamar LLM
            response = self.llm.chat(
                messages=messages, temperature=0.5, max_tokens=4096
            )

            # Verificar si LLM devuelve None (excepcion API o contenido vacio)
            if response is None:
                logger.warning(
                    f"Seccion {section.title} iteracion {iteration + 1}: LLM devolvio None"
                )
                # Si aun hay iteraciones, agregar mensaje y reintentar
                if iteration < max_iterations - 1:
                    messages.append(
                        {"role": "assistant", "content": "(Respuesta vacia)"}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Por favor continue generando contenido.",
                        }
                    )
                    continue
                # Ultima iteracion tambien devuelve None, salir del ciclo y forzar finalizacion
                break

            logger.debug(f"Respuesta LLM: {response[:200]}...")

            # Parsear una vez, reutilizar resultado
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Manejo de conflicto: LLM simultaneamente salida herramientas y Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Seccion {section.title} ronda {iteration + 1}: "
                    f"LLM simultaneamente salida herramientas y Final Answer (conflicto #{conflict_retries})"
                )

                if conflict_retries <= 2:
                    # Primeras dos veces: descartar esta respuesta, pedir a LLM que responda de nuevo
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "【Error de formato】En una respuesta simultaneamente includiste llamada a herramienta y Final Answer, esto no esta permitido.\n"
                                "En cada respuesta solo puedes hacer una de las siguientes dos cosas:\n"
                                "- Llamar una herramienta (output un bloque <tool_call>, no escribir Final Answer)\n"
                                "- Output contenido final (comenzar con 'Final Answer:', no incluir <tool_call>)\n"
                                "Por favor responde de nuevo, solo haz una cosa."
                            ),
                        }
                    )
                    continue
                else:
                    # Tercera vez: degradar, truncar hasta la primera llamada a herramienta, forzar ejecutar
                    logger.warning(
                        f"Seccion {section.title}: {conflict_retries} conflictos consecutivos, "
                        "degradar a truncar y ejecutar primera llamada a herramienta"
                    )
                    first_tool_end = response.find("</tool_call>")
                    if first_tool_end != -1:
                        response = response[: first_tool_end + len("</tool_call>")]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Registrar respuesta LLM
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer,
                )

            # ── Situacion 1: LLM salio Final Answer ──
            if has_final_answer:
                # Llamadas a herramientas insuficientes, rechazar y pedir continuar llamando herramientas
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = (
                        f"(Estas herramientas aun no se han usado, se recomienda usarlas: {', '.join(unused_tools)})"
                        if unused_tools
                        else ""
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                                tool_calls_count=tool_calls_count,
                                min_tool_calls=min_tool_calls,
                                unused_hint=unused_hint,
                            ),
                        }
                    )
                    continue

                # Finalizacion normal
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(
                    f"Seccion {section.title} generada (llamadas a herramientas: {tool_calls_count})"
                )

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count,
                    )
                return final_answer

            # ── Situacion 2: LLM intenta llamar herramienta ──
            if has_tool_calls:
                # Cuota de herramientas agotada → informar explicitamente, pedir output Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append(
                        {
                            "role": "user",
                            "content": REACT_TOOL_LIMIT_MSG.format(
                                tool_calls_count=tool_calls_count,
                                max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                            ),
                        }
                    )
                    continue

                # Solo ejecutar la primera llamada a herramienta
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(
                        f"LLM intenta llamar {len(tool_calls)} herramientas, solo ejecuta la primera: {call['name']}"
                    )

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1,
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
                        iteration=iteration + 1,
                    )

                tool_calls_count += 1
                used_tools.add(call["name"])

                # Construir mensaje de herramientas no usadas
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(
                        unused_list="、".join(unused_tools)
                    )

                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {
                        "role": "user",
                        "content": REACT_OBSERVATION_TEMPLATE.format(
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

            # ── Situacion 3: Ni llamada a herramienta ni Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Llamadas a herramientas insuficientes, recomendar herramientas no usadas
                unused_tools = all_tools - used_tools
                unused_hint = (
                    f"(Estas herramientas aun no se han usado, se recomienda usarlas: {', '.join(unused_tools)})"
                    if unused_tools
                    else ""
                )

                messages.append(
                    {
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    }
                )
                continue

            # Llamadas a herramientas suficientes, LLM salio contenido pero sin prefijo "Final Answer:"
            # Adoptar directamente este contenido como respuesta final, no mas ciclos vacios
            logger.info(
                f"Seccion {section.title} no detecto prefijo 'Final Answer:', adoptar salida LLM directamente como contenido final (llamadas a herramientas: {tool_calls_count})"
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

        # Alcanzar maximo de iteraciones, forzar generacion de contenido
        logger.warning(
            f"Seccion {section.title} alcampo maximo de iteraciones, forzar generacion"
        )
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})

        response = self.llm.chat(messages=messages, temperature=0.5, max_tokens=4096)

        # Verificar si al forzar finalizacion LLM devuelve None
        if response is None:
            logger.error(
                f"Seccion {section.title} al forzar finalizacion LLM devuelve None, usar mensaje de error por defecto"
            )
            final_answer = f"(Esta seccion fallo: LLM devolvio respuesta vacia, por favor intente de nuevo mas tarde)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response

        # Registrar contenido de seccion completado
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
        Generar informe completo (output en tiempo real por secciones)

        Cada seccion generada se guarda inmediatamente en la carpeta, no es necesario esperar a que todo el informe este completo.
        Estructura de archivos:
        reports/{report_id}/
            meta.json       - Metadatos del informe
            outline.json    - Esquema del informe
            progress.json   - Progreso de generacion
            section_01.md   - Seccion 1
            section_02.md   - Seccion 2
            ...
            full_report.md  - Informe completo

        Args:
            progress_callback: Funcion de callback de progreso (stage, progress, message)
            report_id: ID del informe (si no se pasa, se genera automaticamente)

        Returns:
            Report: Informe completo
        """
        import uuid

        # Si no se pasa report_id, generar automaticamente
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()

        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat(),
        )

        # Lista de titulos de secciones completadas (para seguimiento de progreso)
        completed_section_titles = []

        try:
            # Inicializar: crear carpeta de informe y guardar estado inicial
            ReportManager._ensure_report_folder(report_id)

            # Inicializar registrador de logs (logs estructurados agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement,
            )

            # Inicializar registrador de logs de consola (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)

            ReportManager.update_progress(
                report_id,
                "pending",
                0,
                "Inicializando informe...",
                completed_sections=[],
            )
            ReportManager.save_report(report)

            # Fase 1: Planificar esquema
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id,
                "planning",
                5,
                "Comenzando a planificar esquema del informe...",
                completed_sections=[],
            )

            # Registrar inicio de planificacion
            self.report_logger.log_planning_start()

            if progress_callback:
                progress_callback(
                    "planning", 0, "Comenzando a planificar esquema del informe..."
                )

            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: (
                    progress_callback(stage, prog // 5, msg)
                    if progress_callback
                    else None
                )
            )
            report.outline = outline

            # Registrar completado de planificacion
            self.report_logger.log_planning_complete(outline.to_dict())

            # Guardar esquema en archivo
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id,
                "planning",
                15,
                f" Esquema planificado, total {len(outline.sections)} secciones",
                completed_sections=[],
            )
            ReportManager.save_report(report)

            logger.info(f"Esquema guardado en archivo: {report_id}/outline.json")

            # Fase 2: Generar por secciones (guardar por secciones)
            report.status = ReportStatus.GENERATING

            total_sections = len(outline.sections)
            generated_sections = []  # Guardar contenido para contexto

            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)

                # Actualizar progreso
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress,
                    f"Generando seccion: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles,
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        f"Generando seccion: {section.title} ({section_num}/{total_sections})",
                    )

                # Generar contenido principal de la seccion
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

                # Guardar seccion
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Registrar completado de seccion
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip(),
                    )

                logger.info(
                    f"Seccion guardada: {report_id}/section_{section_num:02d}.md"
                )

                # Actualizar progreso
                ReportManager.update_progress(
                    report_id,
                    "generating",
                    base_progress + int(70 / total_sections),
                    f"Seccion {section.title} completada",
                    current_section=None,
                    completed_sections=completed_section_titles,
                )

            # Fase 3: Ensamblar informe completo
            if progress_callback:
                progress_callback("generating", 95, "Ensamblando informe completo...")

            ReportManager.update_progress(
                report_id,
                "generating",
                95,
                "Ensamblando informe completo...",
                completed_sections=completed_section_titles,
            )

            # Usar ReportManager para ensamblar informe completo
            report.markdown_content = ReportManager.assemble_full_report(
                report_id, outline
            )
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()

            # Calcular tiempo total
            total_time_seconds = (datetime.now() - start_time).total_seconds()

            # Registrar completado del informe
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections, total_time_seconds=total_time_seconds
                )

            # Guardar informe final
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id,
                "completed",
                100,
                "Informe generado",
                completed_sections=completed_section_titles,
            )

            if progress_callback:
                progress_callback("completed", 100, "Informe generado")

            logger.info(f"Informe generado: {report_id}")

            # Cerrar registrador de consola
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

        except Exception as e:
            logger.error(f"Error al generar informe: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)

            # Registrar error
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")

            # Guardar estado de fallo
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id,
                    "failed",
                    -1,
                    f"Error al generar informe: {str(e)}",
                    completed_sections=completed_section_titles,
                )
            except Exception:
                pass  # Ignorar errores al guardar

            # Cerrar registrador de consola
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None

            return report

    def chat(
        self, message: str, chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Conversar con Report Agent

        En la conversacion el Agent puede autonomamente llamar herramientas de recuperacion para responder preguntas

        Args:
            message: Mensaje del usuario
            chat_history: Historial de conversacion

        Returns:
            {
                "response": "Respuesta del Agent",
                "tool_calls": [lista de herramientas llamadas],
                "sources": [fuentes de informacion]
            }
        """
        logger.info(f"Conversacion con Report Agent: {message[:50]}...")

        chat_history = chat_history or []

        # Obtener contenido del informe ya generado
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limitar longitud del informe para evitar contexto muy largo
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Contenido del informe truncado] ..."
        except Exception as e:
            logger.warning(f"Error al obtener contenido del informe: {e}")

        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(Aun no hay informe)",
            tools_description=self._get_tools_description(),
        )

        # Construir mensajes
        messages = [{"role": "system", "content": system_prompt}]

        # Agregar historial de conversacion
        for h in chat_history[-10:]:  # Limitar longitud del historial
            messages.append(h)

        # Agregar mensaje del usuario
        messages.append({"role": "user", "content": message})

        # Ciclo ReACT (version simplificada)
        tool_calls_made = []
        max_iterations = 2  # Reducir iteraciones

        for iteration in range(max_iterations):
            response = self.llm.chat(messages=messages, temperature=0.5)

            # Parsear llamadas a herramientas
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # No hay llamada a herramienta, devolver respuesta directamente
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

            # Ejecutar llamadas a herramientas (limitar cantidad)
            tool_results = []
            for call in tool_calls[:1]:  # Maximo 1 llamada a herramienta por ronda
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append(
                    {
                        "tool": call["name"],
                        "result": result[:1500],  # Limitar longitud del resultado
                    }
                )
                tool_calls_made.append(call)

            # Agregar resultados a mensajes
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join(
                [f"[Resultado de {r['tool']}]\n{r['result']}" for r in tool_results]
            )
            messages.append(
                {"role": "user", "content": observation + CHAT_OBSERVATION_SUFFIX}
            )

        # Alcanzar maximo de iteraciones, obtener respuesta final
        final_response = self.llm.chat(messages=messages, temperature=0.5)

        # Limpiar respuesta
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
    Gestor de Informes

    Responsable de la persistencia y recuperacion de informes

    Estructura de archivos (output por secciones):
    reports/
      {report_id}/
        meta.json          - Informacion y estado del informe
        outline.json       - Esquema del informe
        progress.json      - Progreso de generacion
        section_01.md      - Seccion 1
        section_02.md      - Seccion 2
        ...
        full_report.md     - Informe completo
    """

    # Directorio de almacenamiento de informes
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, "reports")

    @classmethod
    def _ensure_reports_dir(cls):
        """Asegurar que exista el directorio raiz de informes"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)

    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Obtener ruta de carpeta del informe"""
        return os.path.join(cls.REPORTS_DIR, report_id)

    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Asegurar que exista la carpeta del informe y devolver ruta"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder

    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Obtener ruta de archivo de metadatos del informe"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")

    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Obtener ruta de archivo Markdown del informe completo"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")

    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Obtener ruta de archivo del esquema"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")

    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Obtener ruta de archivo de progreso"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")

    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Obtener ruta de archivo Markdown de la seccion"""
        return os.path.join(
            cls._get_report_folder(report_id), f"section_{section_index:02d}.md"
        )

    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Obtener ruta de archivo de log del Agent"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")

    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Obtener ruta de archivo de log de consola"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")

    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obtener contenido del log de consola

        Este es el log de salida de consola durante la generacion del informe (INFO, WARNING, etc.),
        diferente de los logs estructurados de agent_log.jsonl.

        Args:
            report_id: ID del informe
            from_line: Desde que linea comenzar a leer (para obtencion incremental, 0 significa desde el principio)

        Returns:
            {
                "logs": [lista de lineas de log],
                "total_lines": total de lineas,
                "from_line": from_line,
                "has_more": si hay mas logs
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
                    # Preservar lineas de log originales, quitar saltos de linea al final
                    logs.append(line.rstrip("\n\r"))

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # Leido hasta el final
        }

    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Obtener log de consola completo (obtener todo de una vez)

        Args:
            report_id: ID del informe

        Returns:
            Lista de lineas de log
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obtener contenido del log del Agent

        Args:
            report_id: ID del informe
            from_line: Desde que linea comenzar a leer (para obtencion incremental, 0 significa desde el principio)

        Returns:
            {
                "logs": [lista de entradas de log],
                "total_lines": total de lineas,
                "from_line": linea de inicio,
                "has_more": si hay mas logs
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
                        # Saltar lineas que fallan en parsing
                        continue

        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False,  # Leido hasta el final
        }

    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obtener log completo del Agent (para obtener todo de una vez)

        Args:
            report_id: ID del informe

        Returns:
            Lista de entradas de log
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]

    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Guardar esquema del informe

        Se llama inmediatamente despues de completar la fase de planificacion
        """
        cls._ensure_report_folder(report_id)

        with open(cls._get_outline_path(report_id), "w", encoding="utf-8") as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Esquema guardado: {report_id}")

    @classmethod
    def save_section(
        cls, report_id: str, section_index: int, section: ReportSection
    ) -> str:
        """
        Guardar una sola seccion

        Se llama inmediatamente despues de generar cada seccion, para output por secciones

        Args:
            report_id: ID del informe
            section_index: Indice de seccion (comienza desde 1)
            section: Objeto de seccion

        Returns:
            Ruta del archivo guardado
        """
        cls._ensure_report_folder(report_id)

        # Construir contenido Markdown de la seccion - limpiar titulos duplicados posibles
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Guardar archivo
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Seccion guardada: {report_id}/{file_suffix}")
        return file_path

    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Limpiar contenido de seccion

        1. Eliminar lineas de titulo Markdown repetidas al inicio con el titulo de la seccion
        2. Convertir todos los titulos de nivel ### y inferior a texto en negrita

        Args:
            content: Contenido original
            section_title: Titulo de la seccion

        Returns:
            Contenido limpio
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

            # Verificar si es una linea de titulo Markdown
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()

                # Verificar si es un titulo repetido con el titulo de la seccion (saltar dentro de las primeras 5 lineas)
                if i < 5:
                    if title_text == section_title or title_text.replace(
                        " ", ""
                    ) == section_title.replace(" ", ""):
                        skip_next_empty = True
                        continue

                # Convertir todos los niveles de titulos (#, ##, ###, ####, etc.) a negrita
                # Porque el titulo de la seccion es agregado por el sistema, el contenido no debe tener ningun titulo
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Agregar linea vacia
                continue

            # Si la linea anterior fue un titulo saltado, y la linea actual esta vacia, tambian saltar
            if skip_next_empty and stripped == "":
                skip_next_empty = False
                continue

            skip_next_empty = False
            cleaned_lines.append(line)

        # Eliminar lineas vacias al inicio
        while cleaned_lines and cleaned_lines[0].strip() == "":
            cleaned_lines.pop(0)

        # Eliminar separadores al inicio
        while cleaned_lines and cleaned_lines[0].strip() in ["---", "***", "___"]:
            cleaned_lines.pop(0)
            # Tambien eliminar lineas vacias despues del separador
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
        Actualizar progreso de generacion del informe

        El frontend puede obtener progreso en tiempo real leyendo progress.json
        """
        cls._ensure_report_folder(report_id)

        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat(),
        }

        with open(cls._get_progress_path(report_id), "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Obtener progreso de generacion del informe"""
        path = cls._get_progress_path(report_id)

        if not os.path.exists(path):
            return None

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obtener lista de secciones ya generadas

        Devuelve toda la informacion de archivos de secciones ya guardadas
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

                # Parsear indice de seccion desde nombre de archivo
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
        Ensamblar informe completo

        Desde archivos de secciones guardados, ensamblar informe completo y limpiar titulos
        """
        folder = cls._get_report_folder(report_id)

        # Construir encabezado del informe
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"

        # Leer todos los archivos de secciones en orden
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]

        # Post-procesamiento: limpiar problemas de titulos en todo el informe
        md_content = cls._post_process_report(md_content, outline)

        # Guardar informe completo
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Informe completo ensamblado: {report_id}")
        return md_content

    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-procesar contenido del informe

        1. Eliminar titulos duplicados
        2. Conservar titulo principal del informe (#) y titulos de seccion (##), eliminar otros niveles (###, ####, etc.)
        3. Limpiar lineas vacias y separadores sobrantes

        Args:
            content: Contenido original del informe
            outline: Esquema del informe

        Returns:
            Contenido procesado
        """
        import re

        lines = content.split("\n")
        processed_lines = []
        prev_was_heading = False

        # Recolectar todos los titulos de seccion del esquema
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Verificar si es linea de titulo
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Verificar si es titulo duplicado (aparecer dentro de 5 lineas consecutivas con mismo contenido)
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
                    # Saltar titulos duplicados y lineas vacias despues
                    i += 1
                    while i < len(lines) and lines[i].strip() == "":
                        i += 1
                    continue

                # Manejo de niveles de titulos:
                # - # (level=1) solo conservar titulo principal del informe
                # - ## (level=2) conservar titulos de secciones
                # - ### y abajo (level>=3) convertir a texto en negrita

                if level == 1:
                    if title == outline.title:
                        # Conservar titulo principal del informe
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Titulo de seccion uso incorrectamente #, corregir a ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Otros titulos de nivel 1 convertir a negrita
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Conservar titulos de secciones
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Titulos de nivel 2 que no son de secciones convertir a negrita
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Titulos de nivel ### y abajo convertir a texto en negrita
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False

                i += 1
                continue

            elif stripped == "---" and prev_was_heading:
                # Saltar separadores justo despues de titulos
                i += 1
                continue

            elif stripped == "" and prev_was_heading:
                # Despues de titulos solo conservar una linea vacia
                if processed_lines and processed_lines[-1].strip() != "":
                    processed_lines.append(line)
                prev_was_heading = False

            else:
                processed_lines.append(line)
                prev_was_heading = False

            i += 1

        # Limpiar varias lineas vacias consecutivas (conservar maximo 2)
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
        """Guardar metadatos del informe e informe completo"""
        cls._ensure_report_folder(report.report_id)

        # Guardar metadatos JSON
        with open(cls._get_report_path(report.report_id), "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # Guardar esquema
        if report.outline:
            cls.save_outline(report.report_id, report.outline)

        # Guardar informe Markdown completo
        if report.markdown_content:
            with open(
                cls._get_report_markdown_path(report.report_id), "w", encoding="utf-8"
            ) as f:
                f.write(report.markdown_content)

        logger.info(f"Informe guardado: {report.report_id}")

    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Obtener informe"""
        path = cls._get_report_path(report_id)

        if not os.path.exists(path):
            # Compatibilidad con formato antiguo: verificar archivo almacenado directamente en directorio reports
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruir objeto Report
        outline = None
        if data.get("outline"):
            outline_data = data["outline"]
            sections = []
            for s in outline_data.get("sections", []):
                sections.append(
                    ReportSection(title=s["title"], content=s.get("content", ""))
                )
            outline = ReportOutline(
                title="Informe de Prediccion Futura",
                summary="Analisis de tendencias futuras y riesgos basado en predicciones de simulacion",
                sections=[
                    ReportSection(title="Escenario de Prediccion y Hallazgos Clave"),
                    ReportSection(
                        title="Analisis de Prediccion de Comportamiento Grupal"
                    ),
                    ReportSection(
                        title="Perspectivas de Tendencias y Avisos de Riesgo"
                    ),
                ],
            )

        # Si markdown_content esta vacio, intentar leer desde full_report.md
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
        """Obtener informe segun ID de simulacion"""
        cls._ensure_reports_dir()

        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Nuevo formato: carpeta
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Compatibilidad con formato antiguo: archivo JSON
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
        """Listar informes"""
        cls._ensure_reports_dir()

        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Nuevo formato: carpeta
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Compatibilidad con formato antiguo: archivo JSON
            elif item.endswith(".json"):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)

        # Ordenar por tiempo de creacion descendente
        reports.sort(key=lambda r: r.created_at, reverse=True)

        return reports[:limit]

    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Eliminar informe (toda la carpeta)"""
        import shutil

        folder_path = cls._get_report_folder(report_id)

        # Nuevo formato: eliminar toda la carpeta
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Carpeta de informe eliminada: {report_id}")
            return True

        # Compatibilidad con formato antiguo: eliminar archivos individuales
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
