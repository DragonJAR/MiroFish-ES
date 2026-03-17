"""
API de rutas de informes
Proporciona interfaces para generación, obtención y diálogo de informes de simulación
"""

import os
import traceback
import threading
from flask import request, jsonify, send_file

from . import report_bp
from ..config import Config
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus
from ..services.simulation_manager import SimulationManager
from ..models.project import ProjectManager
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger

logger = get_logger("mirofish.api.report")


# ============== Interfaz de generación de informe ==============


@report_bp.route("/generate", methods=["POST"])
def generate_report():
    """
    Generar informe de análisis de simulación (tarea asíncrona)

    Esta es una operación que toma tiempo, la interfaz devolverá task_id inmediatamente,
    usa GET /api/report/generate/status para consultar el progreso

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",    // Obligatorio, ID de simulación
            "force_regenerate": false        // Opcional, forzar regeneración
        }

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",
                "status": "generating",
                "message": "Tarea de generación de informe iniciada"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporciona simulation_id"}
            ), 400

        force_regenerate = data.get("force_regenerate", False)

        # Obtener información de simulación
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"La simulación no existe: {simulation_id}"}
            ), 404

        # Verificar si ya existe un informe
        if not force_regenerate:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "report_id": existing_report.report_id,
                            "status": "completed",
                            "message": "El informe ya existe",
                            "already_generated": True,
                        },
                    }
                )

        # Obtener información del proyecto
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify(
                {
                    "success": False,
                    "error": f"El proyecto no existe: {state.project_id}",
                }
            ), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify(
                {
                    "success": False,
                    "error": "Falta el ID del grafo, asegurate de haber construido el grafo",
                }
            ), 400

        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify(
                {
                    "success": False,
                    "error": "Falta la descripción del requerimiento de simulación",
                }
            ), 400

        # Generar report_id de antemano para devolver inmediatamente al frontend
        import uuid

        report_id = f"report_{uuid.uuid4().hex[:12]}"

        # Crear tarea asíncrona
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="report_generate",
            metadata={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "report_id": report_id,
            },
        )

        # Definir tarea en segundo plano con reintentos automáticos
        MAX_RETRIES = 5
        INITIAL_DELAY = 3  # segundos

        def run_generate():
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    task_manager.update_task(
                        task_id,
                        status=TaskStatus.PROCESSING,
                        progress=0,
                        message="Inicializando Report Agent...",
                    )

                    # Crear Report Agent
                    agent = ReportAgent(
                        graph_id=graph_id,
                        simulation_id=simulation_id,
                        simulation_requirement=simulation_requirement,
                    )

                    # Progreso callback
                    def progress_callback(stage, progress, message):
                        task_manager.update_task(
                            task_id, progress=progress, message=f"[{stage}] {message}"
                        )

                    # Generar reporte
                    report = agent.generate_report(
                        progress_callback=progress_callback, report_id=report_id
                    )

                    # Guardar reporte
                    ReportManager.save_report(report)

                    if report.status == ReportStatus.COMPLETED:
                        task_manager.complete_task(
                            task_id,
                            result={
                                "report_id": report.report_id,
                                "simulation_id": simulation_id,
                                "status": "completed",
                            },
                        )
                    else:
                        task_manager.fail_task(
                            task_id, report.error or "Reporte fallido"
                        )

                    # Si llegó aquí, exitosa - salir del loop
                    return

                except Exception as e:
                    error_str = str(e)
                    last_error = e

                    # Verificar si es error 429 (rate limit)
                    is_rate_limit = (
                        "429" in error_str or "rate limit" in error_str.lower()
                    )

                    if is_rate_limit and attempt < MAX_RETRIES - 1:
                        delay = INITIAL_DELAY * (2**attempt)  # Exponential backoff
                        print(
                            f"⚠️ Rate limit detectado (intento {attempt + 1}/{MAX_RETRIES}). Reintentando en {delay}s..."
                        )
                        task_manager.update_task(
                            task_id,
                            progress=-1,
                            message=f"Rate limit, reintento {attempt + 1}/{MAX_RETRIES} en {delay}s...",
                        )
                        import time

                        time.sleep(delay)
                        continue
                    else:
                        # Error no reintenable o se agotaron los reintentos
                        logger.error(f"Reporte falló: {error_str}")
                        task_manager.fail_task(task_id, error_str)
                        return

            # Si salieron del loop por error
            if last_error:
                task_manager.fail_task(
                    task_id,
                    f"Falló después de {MAX_RETRIES} intentos: {str(last_error)}",
                )

        # Iniciar hilo en segundo plano
        thread = threading.Thread(target=run_generate, daemon=True)
        thread.start()

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "report_id": report_id,
                    "task_id": task_id,
                    "status": "generating",
                    "message": "Tarea de generación de informe iniciada, consulta el progreso en /api/report/generate/status",
                    "already_generated": False,
                },
            }
        )

    except Exception as e:
        logger.error(f"Inicio de tarea de generación de informe falló: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/generate/status", methods=["POST"])
def get_generate_status():
    """
    Consultar progreso de tarea de generación de informe

    Solicitud (JSON):
        {
            "task_id": "task_xxxx",         // Opcional, task_id devuelto por generate
            "simulation_id": "sim_xxxx"     // Opcional, ID de simulación
        }

    Retorna:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|failed",
                "progress": 45,
                "message": "..."
            }
        }
    """
    try:
        data = request.get_json() or {}

        task_id = data.get("task_id")
        simulation_id = data.get("simulation_id")

        # Si se proporciona simulation_id, primero verificar si ya existe un informe completado
        if simulation_id:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "report_id": existing_report.report_id,
                            "status": "completed",
                            "progress": 100,
                            "message": "El reporte ya ha sido generado",
                            "already_completed": True,
                        },
                    }
                )

        if not task_id:
            return jsonify(
                {
                    "success": False,
                    "error": "Por favor proporciona task_id o simulation_id",
                }
            ), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            return jsonify(
                {"success": False, "error": f"La tarea no existe: {task_id}"}
            ), 404

        return jsonify({"success": True, "data": task.to_dict()})

    except Exception as e:
        logger.error(f"Consulta de estado de tarea fallida: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Interfaz de obtención de informe ==============


@report_bp.route("/<report_id>", methods=["GET"])
def get_report(report_id: str):
    """
    Obtener detalles del informe

    Retorna:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "simulation_id": "sim_xxxx",
                "status": "completed",
                "outline": {...},
                "markdown_content": "...",
                "created_at": "...",
                "completed_at": "..."
            }
        }
    """
    try:
        report = ReportManager.get_report(report_id)

        if not report:
            return jsonify(
                {"success": False, "error": f"El informe no existe: {report_id}"}
            ), 404

        return jsonify({"success": True, "data": report.to_dict()})

    except Exception as e:
        logger.error(f"Obtención de informe fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/by-simulation/<simulation_id>", methods=["GET"])
def get_report_by_simulation(simulation_id: str):
    """
    Obtener informe según ID de simulación

    Retorna:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                ...
            }
        }
    """
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)

        if not report:
            return jsonify(
                {
                    "success": False,
                    "error": f"No hay informe para esta simulación: {simulation_id}",
                    "has_report": False,
                }
            ), 404

        return jsonify({"success": True, "data": report.to_dict(), "has_report": True})

    except Exception as e:
        logger.error(f"Obtención de informe fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/list", methods=["GET"])
def list_reports():
    """
    Listar todos los informes

    Parámetros de query:
        simulation_id: Filtrar por ID de simulación (opcional)
        limit: Límite de cantidad a devolver (por defecto 50)

    Retorna:
        {
            "success": true,
            "data": [...],
            "count": 10
        }
    """
    try:
        simulation_id = request.args.get("simulation_id")
        limit = request.args.get("limit", 50, type=int)

        reports = ReportManager.list_reports(simulation_id=simulation_id, limit=limit)

        return jsonify(
            {
                "success": True,
                "data": [r.to_dict() for r in reports],
                "count": len(reports),
            }
        )

    except Exception as e:
        logger.error(f"Listar informes fallido: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/download", methods=["GET"])
def download_report(report_id: str):
    """
    Descargar informe (formato Markdown)

    Retorna archivo Markdown
    """
    try:
        report = ReportManager.get_report(report_id)

        if not report:
            return jsonify(
                {"success": False, "error": f"El informe no existe: {report_id}"}
            ), 404

        md_path = ReportManager._get_report_markdown_path(report_id)

        if not os.path.exists(md_path):
            # Si el archivo MD no existe, generar un archivo temporal
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name

            return send_file(
                temp_path, as_attachment=True, download_name=f"{report_id}.md"
            )

        return send_file(md_path, as_attachment=True, download_name=f"{report_id}.md")

    except Exception as e:
        logger.error(f"Descarga de informe fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>", methods=["DELETE"])
def delete_report(report_id: str):
    """Eliminar informe"""
    try:
        success = ReportManager.delete_report(report_id)

        if not success:
            return jsonify(
                {"success": False, "error": f"El informe no existe: {report_id}"}
            ), 404

        return jsonify({"success": True, "message": f"Informe eliminado: {report_id}"})

    except Exception as e:
        logger.error(f"Eliminación de informe fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de diálogo con Report Agent ==============


@report_bp.route("/chat", methods=["POST"])
def chat_with_report_agent():
    """
    Dialogar con Report Agent

    Report Agent puede llamar herramientas de recuperación de forma autónoma durante el diálogo

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",        // Obligatorio, ID de simulación
            "message": "Por favor explica la tendencia del sentimiento",    // Obligatorio, mensaje del usuario
            "chat_history": [                   // Opcional, historial de diálogo
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

    Retorna:
        {
            "success": true,
            "data": {
                "response": "Respuesta del Agent...",
                "tool_calls": [lista de herramientas llamadas],
                "sources": [fuentes de información]
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        message = data.get("message")
        chat_history = data.get("chat_history", [])

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporciona simulation_id"}
            ), 400

        if not message:
            return jsonify(
                {"success": False, "error": "Por favor proporciona message"}
            ), 400

        # Obtener información de simulación y proyecto
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"La simulación no existe: {simulation_id}"}
            ), 404

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify(
                {
                    "success": False,
                    "error": f"El proyecto no existe: {state.project_id}",
                }
            ), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": "Falta el ID del grafo"}), 400

        simulation_requirement = project.simulation_requirement or ""

        # Crear agente y mantener conversación
        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement,
        )

        result = agent.chat(message=message, chat_history=chat_history)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Diálogo fallido: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de progreso y secciones del informe ==============


@report_bp.route("/<report_id>/progress", methods=["GET"])
def get_report_progress(report_id: str):
    """
    Obtener progreso de generación del informe (tiempo real)

    Retorna:
        {
            "success": true,
            "data": {
                "status": "generating",
                "progress": 45,
                "message": "Generando sección: Hallazgos clave",
                "current_section": "Hallazgos clave",
                "completed_sections": ["Resumen ejecutivo", "Contexto de simulación"],
                "updated_at": "2025-12-09T..."
            }
        }
    """
    try:
        progress = ReportManager.get_progress(report_id)

        if not progress:
            return jsonify(
                {
                    "success": False,
                    "error": f"El informe no existe o información de progreso no disponible: {report_id}",
                }
            ), 404

        return jsonify({"success": True, "data": progress})

    except Exception as e:
        logger.error(f"Obtención de progreso del informe fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/sections", methods=["GET"])
def get_report_sections(report_id: str):
    """
    Obtener lista de secciones ya generadas (salida por secciones)

    El frontend puede consultar esta interfaz para obtener el contenido de las secciones generadas
    sin esperar a que todo el informe esté completo

    Retorna:
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "sections": [
                    {
                        "filename": "section_01.md",
                        "section_index": 1,
                        "content": "## Resumen ejecutivo\\n\\n..."
                    },
                    ...
                ],
                "total_sections": 3,
                "is_complete": false
            }
        }
    """
    try:
        sections = ReportManager.get_generated_sections(report_id)

        # Obtener estado del informe
        report = ReportManager.get_report(report_id)
        is_complete = report is not None and report.status == ReportStatus.COMPLETED

        return jsonify(
            {
                "success": True,
                "data": {
                    "report_id": report_id,
                    "sections": sections,
                    "total_sections": len(sections),
                    "is_complete": is_complete,
                },
            }
        )

    except Exception as e:
        logger.error(f"Obtención de lista de secciones fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/section/<int:section_index>", methods=["GET"])
def get_single_section(report_id: str, section_index: int):
    """
    Obtener contenido de una sola sección

    Retorna:
        {
            "success": true,
            "data": {
                "filename": "section_01.md",
                "content": "## Resumen ejecutivo\\n\\n..."
            }
        }
    """
    try:
        section_path = ReportManager._get_section_path(report_id, section_index)

        if not os.path.exists(section_path):
            return jsonify(
                {
                    "success": False,
                    "error": f"La sección no existe: section_{section_index:02d}.md",
                }
            ), 404

        with open(section_path, "r", encoding="utf-8") as f:
            content = f.read()

        return jsonify(
            {
                "success": True,
                "data": {
                    "filename": f"section_{section_index:02d}.md",
                    "section_index": section_index,
                    "content": content,
                },
            }
        )

    except Exception as e:
        logger.error(f"Obtención de contenido de sección fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de verificación de estado del informe ==============


@report_bp.route("/check/<simulation_id>", methods=["GET"])
def check_report_status(simulation_id: str):
    """
    Verificar si la simulación tiene informe y el estado del informe

    Usado por el frontend para determinar si desbloquear la funcionalidad Interview

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "has_report": true,
                "report_status": "completed",
                "report_id": "report_xxxx",
                "interview_unlocked": true
            }
        }
    """
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)

        has_report = report is not None
        report_status = report.status.value if report else None
        report_id = report.report_id if report else None

        # Solo se desbloquea interview cuando el informe está completo
        interview_unlocked = has_report and report.status == ReportStatus.COMPLETED

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "has_report": has_report,
                    "report_status": report_status,
                    "report_id": report_id,
                    "interview_unlocked": interview_unlocked,
                },
            }
        )

    except Exception as e:
        logger.error(f"Verificación de estado del informe fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de logs del Agent ==============


@report_bp.route("/<report_id>/agent-log", methods=["GET"])
def get_agent_log(report_id: str):
    """
    Obtener logs detallados de ejecución del Report Agent

    Obtiene en tiempo real cada paso del proceso de generación del informe, incluyendo:
    - Inicio del informe, inicio/completado de planificación
    - Inicio de cada sección, llamadas a herramientas, respuestas LLM, completado
    - Informe completado o fallido

    Parámetros de query:
        from_line: Desde qué línea empezar a leer (opcional, por defecto 0, para obtención incremental)

    Retorna:
        {
            "success": true,
            "data": {
                "logs": [
                    {
                        "timestamp": "2025-12-13T...",
                        "elapsed_seconds": 12.5,
                        "report_id": "report_xxxx",
                        "action": "tool_call",
                        "stage": "generating",
                        "section_title": "Resumen ejecutivo",
                        "section_index": 1,
                        "details": {
                            "tool_name": "insight_forge",
                            "parameters": {...},
                            ...
                        }
                    },
                    ...
                ],
                "total_lines": 25,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        from_line = request.args.get("from_line", 0, type=int)

        log_data = ReportManager.get_agent_log(report_id, from_line=from_line)

        return jsonify({"success": True, "data": log_data})

    except Exception as e:
        logger.error(f"Obtención de logs del Agent fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/agent-log/stream", methods=["GET"])
def stream_agent_log(report_id: str):
    """
    Obtener logs completos del Agent (todos de una vez)

    Retorna:
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 25
            }
        }
    """
    try:
        logs = ReportManager.get_agent_log_stream(report_id)

        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})

    except Exception as e:
        logger.error(f"Obtención de logs del Agent fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de logs de consola ==============


@report_bp.route("/<report_id>/console-log", methods=["GET"])
def get_console_log(report_id: str):
    """
    Obtener logs de salida de consola del Report Agent

    Obtiene en tiempo real la salida de consola durante la generación del informe (INFO, WARNING, etc.),
    esto es diferente de los logs JSON estructurados devueltos por la interfaz agent-log,
    es un log de estilo consola en formato de texto plano.

    Parámetros de query:
        from_line: Desde qué línea empezar a leer (opcional, por defecto 0, para obtención incremental)

    Retorna:
        {
            "success": true,
            "data": {
                "logs": [
                    "[19:46:14] INFO: Búsqueda completada: Se encontraron 15 hechos relacionados",
                    "[19:46:14] INFO: Búsqueda en grafo: graph_id=xxx, query=...",
                    ...
                ],
                "total_lines": 100,
                "from_line": 0,
                "has_more": false
            }
        }
    """
    try:
        from_line = request.args.get("from_line", 0, type=int)

        log_data = ReportManager.get_console_log(report_id, from_line=from_line)

        return jsonify({"success": True, "data": log_data})

    except Exception as e:
        logger.error(f"Obtención de logs de consola fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/console-log/stream", methods=["GET"])
def stream_console_log(report_id: str):
    """
    Obtener logs completos de consola (todos de una vez)

    Retorna:
        {
            "success": true,
            "data": {
                "logs": [...],
                "count": 100
            }
        }
    """
    try:
        logs = ReportManager.get_console_log_stream(report_id)

        return jsonify({"success": True, "data": {"logs": logs, "count": len(logs)}})

    except Exception as e:
        logger.error(f"Obtención de logs de consola fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de herramientas (para depuración) ==============


@report_bp.route("/tools/search", methods=["POST"])
def search_graph_tool():
    """
    Interfaz de herramienta de búsqueda en grafo (para depuración)

    Solicitud (JSON):
        {
            "graph_id": "mirofish_xxxx",
            "query": "Consulta de búsqueda",
            "limit": 10
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")
        query = data.get("query")
        limit = data.get("limit", 10)

        if not graph_id or not query:
            return jsonify(
                {"success": False, "error": "Por favor proporciona graph_id y query"}
            ), 400

        from ..services.zep_tools import ZepToolsService

        tools = ZepToolsService()
        result = tools.search_graph(graph_id=graph_id, query=query, limit=limit)

        return jsonify({"success": True, "data": result.to_dict()})

    except Exception as e:
        logger.error(f"Búsqueda en grafo fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/tools/statistics", methods=["POST"])
def get_graph_statistics_tool():
    """
    Interfaz de herramienta de estadísticas del grafo (para depuración)

    Solicitud (JSON):
        {
            "graph_id": "mirofish_xxxx"
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")

        if not graph_id:
            return jsonify(
                {"success": False, "error": "Por favor proporciona graph_id"}
            ), 400

        from ..services.zep_tools import ZepToolsService

        tools = ZepToolsService()
        result = tools.get_graph_statistics(graph_id)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Obtención de estadísticas del grafo fallida: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500
