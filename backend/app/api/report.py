"""
API de rutas de reportes
Proporciona interfaces para generación, obtención y diálogo de reportes de simulación
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
from ..utils.locale import t, get_locale, set_locale

logger = get_logger("mirofish.api.report")


# ============== Interfaces de generación de reportes ==============


@report_bp.route("/generate", methods=["POST"])
def generate_report():
    """
    Generar reporte de análisis de simulación (tarea asíncrona)

    Esta es una operación que consume tiempo, la interfaz retorna task_id inmediatamente,
    use GET /api/report/generate/status para consultar el progreso

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",    // Obligatorio, ID de simulación
            "force_regenerate": false        // Opcional, forzar regeneración
        }

    Respuesta:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",
                "status": "generating",
                "message": "Tarea de generación de reporte iniciada"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        force_regenerate = data.get("force_regenerate", False)

        # Obtener información de simulación
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {
                    "success": False,
                    "error": t("api.simulationNotFound", id=simulation_id),
                }
            ), 404

        # Verificar si ya existe reporte
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
                            "message": t("api.reportAlreadyExists"),
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
                    "error": t("api.projectNotFound", id=state.project_id),
                }
            ), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify(
                {"success": False, "error": t("api.missingGraphIdEnsure")}
            ), 400

        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify(
                {"success": False, "error": t("api.missingSimRequirement")}
            ), 400

        # Generar report_id por adelantado, para retornar inmediatamente al frontend
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

        # Capturar locale antes de iniciar hilo en background
        current_locale = get_locale()

        # Definir tarea de background
        def run_generate():
            set_locale(current_locale)
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message=t("api.initReportAgent"),
                )

                # Crear Report Agent
                agent = ReportAgent(
                    graph_id=graph_id,
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                )

                # Callback de progreso
                def progress_callback(stage, progress, message):
                    task_manager.update_task(
                        task_id, progress=progress, message=f"[{stage}] {message}"
                    )

                # Generar reporte (pasando report_id pre-generado)
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
                        task_id, report.error or t("api.reportGenerateFailed")
                    )

            except Exception as e:
                logger.error(f"Fallo de generación de informe: {str(e)}")
                task_manager.fail_task(task_id, str(e))

        # Iniciar hilo de background
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
                    "message": t("api.reportGenerateStarted"),
                    "already_generated": False,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error al iniciar tarea de generación de reporte: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/generate/status", methods=["POST"])
def get_generate_status():
    """
    Consultar progreso de tarea de generación de reporte

    Solicitud (JSON):
        {
            "task_id": "task_xxxx",         // Opcional, task_id retornado por generate
            "simulation_id": "sim_xxxx"     // Opcional, ID de simulación
        }

    Respuesta:
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

        # Si se proporciona simulation_id, verificar primero si ya existe reporte completado
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
                            "message": t("api.reportGenerated"),
                            "already_completed": True,
                        },
                    }
                )

        if not task_id:
            return jsonify(
                {"success": False, "error": t("api.requireTaskOrSimId")}
            ), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            return jsonify(
                {"success": False, "error": t("api.taskNotFound", id=task_id)}
            ), 404

        return jsonify({"success": True, "data": task.to_dict()})

    except Exception as e:
        logger.error(f"Error al consultar estado de tarea: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# ============== Interfaces de obtención de reportes ==============


@report_bp.route("/<report_id>", methods=["GET"])
def get_report(report_id: str):
    """
    Obtener detalles del reporte

    Respuesta:
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
                {"success": False, "error": t("api.reportNotFound", id=report_id)}
            ), 404

        return jsonify({"success": True, "data": report.to_dict()})

    except Exception as e:
        logger.error(f"Error al obtener reporte: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/by-simulation/<simulation_id>", methods=["GET"])
def get_report_by_simulation(simulation_id: str):
    """
    Obtener reporte por ID de simulación

    Respuesta:
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
                    "error": t("api.noReportForSim", id=simulation_id),
                    "has_report": False,
                }
            ), 404

        return jsonify({"success": True, "data": report.to_dict(), "has_report": True})

    except Exception as e:
        logger.error(f"Error al obtener reporte: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/list", methods=["GET"])
def list_reports():
    """
    Listar todos los reportes

    Parámetros Query:
        simulation_id: Filtrar por ID de simulación (opcional)
        limit: Límite de cantidad retornada (por defecto 50)

    Respuesta:
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
        logger.error(f"Error al listar reportes: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/download", methods=["GET"])
def download_report(report_id: str):
    """
    Descargar reporte (formato Markdown)

    Retorna archivo Markdown
    """
    try:
        report = ReportManager.get_report(report_id)

        if not report:
            return jsonify(
                {"success": False, "error": t("api.reportNotFound", id=report_id)}
            ), 404

        md_path = ReportManager._get_report_markdown_path(report_id)

        if not os.path.exists(md_path):
            # Si archivo MD no existe, generar un archivo temporal
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name

            return send_file(
                temp_path, as_attachment=True, download_name=f"{report_id}.md"
            )

        return send_file(md_path, as_attachment=True, download_name=f"{report_id}.md")

    except Exception as e:
        logger.error(f"Error al descargar reporte: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>", methods=["DELETE"])
def delete_report(report_id: str):
    """Eliminar reporte"""
    try:
        success = ReportManager.delete_report(report_id)

        if not success:
            return jsonify(
                {"success": False, "error": t("api.reportNotFound", id=report_id)}
            ), 404

        return jsonify(
            {"success": True, "message": t("api.reportDeleted", id=report_id)}
        )

    except Exception as e:
        logger.error(f"Error al eliminar reporte: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaces de diálogo con Report Agent ==============


@report_bp.route("/chat", methods=["POST"])
def chat_with_report_agent():
    """
    Dialogar con Report Agent

    El Report Agent puede invocar herramientas de recuperación autónomamente durante el diálogo para responder preguntas

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",        // Obligatorio, ID de simulación
            "message": "Por favor explica la tendencia de opinión pública",    // Obligatorio, mensaje del usuario
            "chat_history": [                   // Opcional, historial de diálogo
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

    Respuesta:
        {
            "success": true,
            "data": {
                "response": "Respuesta del Agent...",
                "tool_calls": [lista de herramientas invocadas],
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
                {"success": False, "error": t("api.requireSimulationId")}
            ), 400

        if not message:
            return jsonify({"success": False, "error": t("api.requireMessage")}), 400

        # Obtener simulación e información del proyecto
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {
                    "success": False,
                    "error": t("api.simulationNotFound", id=simulation_id),
                }
            ), 404

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify(
                {
                    "success": False,
                    "error": t("api.projectNotFound", id=state.project_id),
                }
            ), 404

        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({"success": False, "error": t("api.missingGraphId")}), 400

        simulation_requirement = project.simulation_requirement or ""

        # Crear Agent y realizar diálogo
        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement,
        )

        result = agent.chat(message=message, chat_history=chat_history)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Error en diálogo: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaces de progreso de reporte y secciones ==============


@report_bp.route("/<report_id>/progress", methods=["GET"])
def get_report_progress(report_id: str):
    """
    Obtener progreso de generación de reporte (tiempo real)

    Respuesta:
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
                    "error": t("api.reportProgressNotAvail", id=report_id),
                }
            ), 404

        return jsonify({"success": True, "data": progress})

    except Exception as e:
        logger.error(f"Error al obtener progreso de reporte: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/sections", methods=["GET"])
def get_report_sections(report_id: str):
    """
    Obtener lista de secciones ya generadas (salida por secciones)

    El frontend puede hacer polling de esta interfaz para obtener contenido de secciones ya generadas, sin necesidad de esperar que todo el reporte se complete

    Respuesta:
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

        # Obtener estado del reporte
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
        logger.error(f"Error al obtener lista de capítulos: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/section/<int:section_index>", methods=["GET"])
def get_single_section(report_id: str, section_index: int):
    """
    Obtener contenido de una sección individual

    Respuesta:
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
                    "error": t("api.sectionNotFound", index=f"{section_index:02d}"),
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
        logger.error(f"Error al obtener contenido de capítulo: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaces de verificación de estado de reporte ==============


@report_bp.route("/check/<simulation_id>", methods=["GET"])
def check_report_status(simulation_id: str):
    """
    Verificar si la simulación tiene reporte, y el estado del reporte

    Utilizado por el frontend para determinar si desbloquear funcionalidad de Interview

    Respuesta:
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

        # Solo desbloquear interview cuando el reporte esté completado
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
        logger.error(f"Error al verificar estado de reporte: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaces de logs de Agent ==============


@report_bp.route("/<report_id>/agent-log", methods=["GET"])
def get_agent_log(report_id: str):
    """
    Obtener logs de ejecución detallados del Report Agent

    Obtener en tiempo real cada paso de acción durante el proceso de generación de reporte, incluyendo:
    - Inicio del reporte, inicio/completado de planificación
    - Inicio de cada sección, llamada a herramientas, respuesta LLM, completado
    - Completado o fallido del reporte

    Parámetros Query:
        from_line: Comenzar lectura desde qué línea (opcional, por defecto 0, para obtención incremental)

    Respuesta:
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
        logger.error(f"Error al obtener logs del agente: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/agent-log/stream", methods=["GET"])
def stream_agent_log(report_id: str):
    """
    Obtener logs completos de Agent (obtener todo de una vez)

    Respuesta:
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
        logger.error(f"Error al obtener logs del agente: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaces de logs de consola ==============


@report_bp.route("/<report_id>/console-log", methods=["GET"])
def get_console_log(report_id: str):
    """
    Obtener logs de salida de consola del Report Agent

    Obtener en tiempo real la salida de consola durante el proceso de generación de reporte (INFO, WARNING, etc),
    Esto es diferente de los logs JSON estructurados retornados por la interfaz agent-log,
    es logs de estilo de consola en formato de texto plano.

    Parámetros Query:
        from_line: Comenzar lectura desde qué línea (opcional, por defecto 0, para obtención incremental)

    Respuesta:
        {
            "success": true,
            "data": {
                "logs": [
                    "[19:46:14] INFO: Búsqueda completada: Se encontraron 15 hechos relevantes",
                    "[19:46:14] INFO: Búsqueda de grafo: graph_id=xxx, query=...",
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
        logger.error(f"Error al obtener logs de consola: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/<report_id>/console-log/stream", methods=["GET"])
def stream_console_log(report_id: str):
    """
    Obtener logs completos de consola (obtener todo de una vez)

    Respuesta:
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
        logger.error(f"Error al obtener logs de consola: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaces de llamadas a herramientas (para uso de depuración) ==============


@report_bp.route("/tools/search", methods=["POST"])
def search_graph_tool():
    """
    Interfaz de herramienta de búsqueda de grafo (para uso de depuración)

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
                {"success": False, "error": t("api.requireGraphIdAndQuery")}
            ), 400

        from ..services.zep_tools import ZepToolsService

        tools = ZepToolsService()
        result = tools.search_graph(graph_id=graph_id, query=query, limit=limit)

        return jsonify({"success": True, "data": result.to_dict()})

    except Exception as e:
        logger.error(f"Error en búsqueda de grafo: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@report_bp.route("/tools/statistics", methods=["POST"])
def get_graph_statistics_tool():
    """
    Interfaz de herramienta de estadísticas de grafo (para uso de depuración)

    Solicitud (JSON):
        {
            "graph_id": "mirofish_xxxx"
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")

        if not graph_id:
            return jsonify({"success": False, "error": t("api.requireGraphId")}), 400

        from ..services.zep_tools import ZepToolsService

        tools = ZepToolsService()
        result = tools.get_graph_statistics(graph_id)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Error al obtener estadísticas de grafo: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500
