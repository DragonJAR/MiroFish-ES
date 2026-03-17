"""
API de Simulacion
Step2: Lectura de entidades Zep y filtrado, Preparacion y ejecucion de simulacion OASIS (todo automatizado)
"""

import os
import traceback
from flask import request, jsonify, send_file

from . import simulation_bp
from ..config import Config
from ..services.zep_entity_reader import ZepEntityReader
from ..services.oasis_profile_generator import OasisProfileGenerator
from ..services.simulation_manager import SimulationManager, SimulationStatus
from ..services.simulation_runner import SimulationRunner, RunnerStatus
from ..utils.logger import get_logger
from ..models.project import ProjectManager

logger = get_logger("mirofish.api.simulation")


# Interview prompt optimization prefix
# Adding this prefix can avoid the Agent calling tools, and directly reply with text
INTERVIEW_PROMPT_PREFIX = "Combinando tu personalidad, todos tus recuerdos y acciones pasados, responde directamente con texto sin llamar a ninguna herramienta:"


def optimize_interview_prompt(prompt: str) -> str:
    """
    Optimizar pregunta de entrevista, agregar prefijo para evitar que Agent llame herramientas

    Args:
        prompt: Pregunta original

    Returns:
        Pregunta optimizada
    """
    if not prompt:
        return prompt
    # Evitar agregar prefijo duplicado
    if prompt.startswith(INTERVIEW_PROMPT_PREFIX):
        return prompt
    return f"{INTERVIEW_PROMPT_PREFIX}{prompt}"


# ============== Interfaces de lectura de entidades ==============


@simulation_bp.route("/entities/<graph_id>", methods=["GET"])
def get_graph_entities(graph_id: str):
    """
    Obtener todas las entidades del grafo (filtradas)

    Solo devuelve nodos que coinciden con tipos de entidad predefinidos (nodos con Labels que no son solo Entity)

    Query parametros:
        entity_types: Lista de tipos de entidad separados por coma (opcional, para filtrar mas)
        enrich: Si obtener informacion de bordes relacionados (default true)
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify(
                {"success": False, "error": "ZEP_API_KEY no configurada"}
            ), 500

        entity_types_str = request.args.get("entity_types", "")
        entity_types = (
            [t.strip() for t in entity_types_str.split(",") if t.strip()]
            if entity_types_str
            else None
        )
        enrich = request.args.get("enrich", "true").lower() == "true"

        logger.info(
            f"Obtener entidades del grafo: graph_id={graph_id}, entity_types={entity_types}, enrich={enrich}"
        )

        reader = ZepEntityReader()
        result = reader.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=entity_types,
            enrich_with_edges=enrich,
        )

        return jsonify({"success": True, "data": result.to_dict()})

    except Exception as e:
        logger.error(f"Error al obtener entidades del grafo: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/entities/<graph_id>/<entity_uuid>", methods=["GET"])
def get_entity_detail(graph_id: str, entity_uuid: str):
    """Obtener detalles de una sola entidad"""
    try:
        if not Config.ZEP_API_KEY:
            return jsonify(
                {"success": False, "error": "ZEP_API_KEY no configurada"}
            ), 500

        reader = ZepEntityReader()
        entity = reader.get_entity_with_context(graph_id, entity_uuid)

        if not entity:
            return jsonify(
                {"success": False, "error": f"La entidad no existe: {entity_uuid}"}
            ), 404

        return jsonify({"success": True, "data": entity.to_dict()})

    except Exception as e:
        logger.error(f"Error al obtener detalles de entidad: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/entities/<graph_id>/by-type/<entity_type>", methods=["GET"])
def get_entities_by_type(graph_id: str, entity_type: str):
    """Obtener todas las entidades de un tipo especificado"""
    try:
        if not Config.ZEP_API_KEY:
            return jsonify(
                {"success": False, "error": "ZEP_API_KEY no configurada"}
            ), 500

        enrich = request.args.get("enrich", "true").lower() == "true"

        reader = ZepEntityReader()
        entities = reader.get_entities_by_type(
            graph_id=graph_id, entity_type=entity_type, enrich_with_edges=enrich
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "entity_type": entity_type,
                    "count": len(entities),
                    "entities": [e.to_dict() for e in entities],
                },
            }
        )

    except Exception as e:
        logger.error(f"Error al obtener entidades: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaces de gestion de simulacion ==============


@simulation_bp.route("/create", methods=["POST"])
def create_simulation():
    """
    Crear nueva simulacion

    Nota: parametros como max_rounds son generados inteligente por LLM, no requieren configuracion manual

    Request (JSON):
        {
            "project_id": "proj_xxxx",      // Requerido
            "graph_id": "mirofish_xxxx",    // Opcional, si no se proporciona se obtiene del proyecto
            "enable_twitter": true,          // Opcional, por defecto true
            "enable_reddit": true            // Opcional, por defecto true
        }

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "project_id": "proj_xxxx",
                "graph_id": "mirofish_xxxx",
                "status": "created",
                "enable_twitter": true,
                "enable_reddit": true,
                "created_at": "2025-12-01T10:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}

        project_id = data.get("project_id")
        if not project_id:
            return jsonify(
                {"success": False, "error": "Por favor proporciona project_id"}
            ), 400

        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify(
                {"success": False, "error": f"El proyecto no existe: {project_id}"}
            ), 404

        graph_id = data.get("graph_id") or project.graph_id
        if not graph_id:
            return jsonify(
                {
                    "success": False,
                    "error": "El proyecto aun no ha construido el grafo, por favor llama primero a /api/graph/build",
                }
            ), 400

        manager = SimulationManager()
        state = manager.create_simulation(
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get("enable_twitter", True),
            enable_reddit=data.get("enable_reddit", True),
        )

        return jsonify({"success": True, "data": state.to_dict()})

    except Exception as e:
        logger.error(f"Error al crear simulacion: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


def _check_simulation_prepared(simulation_id: str) -> tuple:
    """
    Verificar si la simulacion ya esta preparada

    Condiciones de verificacion:
    1. state.json existe y status es "ready"
    2. Archivos necesarios existen: reddit_profiles.json, twitter_profiles.csv, simulation_config.json

    Nota: Los scripts de ejecucion (run_*.py) se mantienen en el directorio backend/scripts/, ya no se copian al directorio de simulacion

    Args:
        simulation_id: ID de simulacion

    Returns:
        (is_prepared: bool, info: dict)
    """
    import os
    from ..config import Config

    simulation_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

    # Verificar si el directorio existe
    if not os.path.exists(simulation_dir):
        return False, {"reason": "El directorio de simulacion no existe"}

    # Lista de archivos necesarios (no incluye scripts, los scripts estan en backend/scripts/)
    required_files = [
        "state.json",
        "simulation_config.json",
        "reddit_profiles.json",
        "twitter_profiles.csv",
    ]

    # Verificar si los archivos existen
    existing_files = []
    missing_files = []
    for f in required_files:
        file_path = os.path.join(simulation_dir, f)
        if os.path.exists(file_path):
            existing_files.append(f)
        else:
            missing_files.append(f)

    if missing_files:
        return False, {
            "reason": "Faltan archivos necesarios",
            "missing_files": missing_files,
            "existing_files": existing_files,
        }

    # Verificar el estado en state.json
    state_file = os.path.join(simulation_dir, "state.json")
    try:
        import json

        with open(state_file, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        status = state_data.get("status", "")
        config_generated = state_data.get("config_generated", False)

        # Log detallado
        logger.debug(
            f"Verificando estado de preparacion de simulacion: {simulation_id}, status={status}, config_generated={config_generated}"
        )

        # Si config_generated=True y los archivos existen, se considera preparado
        # Los siguientes estados indican que la preparacion ya esta completa:
        # - ready: preparado, puede ejecutar
        # - preparing: si config_generated=True indica que ya termino
        # - running: esta ejecutando, la preparacion ya estaba completa
        # - completed: ejecucion completada, la preparacion ya estaba completa
        # - stopped: detenido, la preparacion ya estaba completa
        # - failed: fallo en ejecucion (pero la preparacion esta completa)
        prepared_statuses = [
            "ready",
            "preparing",
            "running",
            "completed",
            "stopped",
            "failed",
        ]
        if status in prepared_statuses and config_generated:
            # Obtener informacion de estadisticas de archivos
            profiles_file = os.path.join(simulation_dir, "reddit_profiles.json")
            config_file = os.path.join(simulation_dir, "simulation_config.json")

            profiles_count = 0
            if os.path.exists(profiles_file):
                with open(profiles_file, "r", encoding="utf-8") as f:
                    profiles_data = json.load(f)
                    profiles_count = (
                        len(profiles_data) if isinstance(profiles_data, list) else 0
                    )

            # Si el estado es preparing pero los archivos ya estan completos, actualizar automaticamente a ready
            if status == "preparing":
                try:
                    state_data["status"] = "ready"
                    from datetime import datetime

                    state_data["updated_at"] = datetime.now().isoformat()
                    with open(state_file, "w", encoding="utf-8") as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"Actualizacion automatica de estado de simulacion: {simulation_id} preparing -> ready"
                    )
                    status = "ready"
                except Exception as e:
                    logger.warning(f"Error al actualizar estado automaticamente: {e}")

            logger.info(
                f"Resultado de verificacion de simulacion {simulation_id}: Preparada (status={status}, config_generated={config_generated})"
            )
            return True, {
                "status": status,
                "entities_count": state_data.get("entities_count", 0),
                "profiles_count": profiles_count,
                "entity_types": state_data.get("entity_types", []),
                "config_generated": config_generated,
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "existing_files": existing_files,
            }
        else:
            logger.warning(
                f"Resultado de verificacion de simulacion {simulation_id}: No preparada (status={status}, config_generated={config_generated})"
            )
            return False, {
                "reason": f"El estado no esta en la lista de preparados o config_generated es false: status={status}, config_generated={config_generated}",
                "status": status,
                "config_generated": config_generated,
            }

    except Exception as e:
        return False, {"reason": f"Error al leer archivo de estado: {str(e)}"}


@simulation_bp.route("/prepare", methods=["POST"])
def prepare_simulation():
    """
    Preparar entorno de simulacion (tarea asincronica, LLM genera todos los parametros inteligentemente)

    Esta es una operacion que toma tiempo, la interfaz retornara task_id inmediatamente,
    usa GET /api/simulation/prepare/status para consultar el progreso

    Caracteristicas:
    - Detectar automaticamente preparaciones completadas, evitar regeneracion duplicada
    - Si ya esta preparado, retornar resultados existentes directamente
    - Soportar regeneracion forzada (force_regenerate=true)

    Pasos:
    1. Verificar si ya existe preparacion completada
    2. Leer y filtrar entidades del grafo Zep
    3. Generar OASIS Agent Profile para cada entidad (con mecanismo de reintento)
    4. LLM genera configuracion de simulacion inteligentemente (con mecanismo de reintento)
    5. Guardar archivos de configuracion y scripts preset

    Request (JSON):
        {
            "simulation_id": "sim_xxxx",                   // Requerido, ID de simulacion
            "entity_types": ["Student", "PublicFigure"],  // Opcional, especificar tipos de entidad
            "use_llm_for_profiles": true,                 // Opcional, si usar LLM para generar perfiles
            "parallel_profile_count": 5,                  // Opcional, cantidad de perfiles paralelos, por defecto 5
            "force_regenerate": false                     // Opcional, forzar regeneracion, por defecto false
        }

    Returns:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",           // Retorna cuando es nueva tarea
                "status": "preparing|ready",
                "message": "Tarea de preparacion iniciada|Ya existe preparacion completada",
                "already_prepared": true|false    // Si ya esta preparado
            }
        }
    """
    import threading
    import os
    from ..models.task import TaskManager, TaskStatus
    from ..config import Config

    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"La simulación no existe: {simulation_id}"}
            ), 404

        # Verificar si se fuerza regenerar
        force_regenerate = data.get("force_regenerate", False)
        logger.info(
            f"Iniciando procesamiento de solicitud /prepare: simulation_id={simulation_id}, force_regenerate={force_regenerate}"
        )

        # Verificar si ya está preparado (evitar generación duplicada)
        if not force_regenerate:
            logger.debug(f"Verificar simulación {simulation_id} ya está preparado...")
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            logger.debug(
                f"Resultado de verificación: is_prepared={is_prepared}, prepare_info={prepare_info}"
            )
            if is_prepared:
                logger.info(
                    f"Simulación {simulation_id} ya está preparado，omitir generación duplicada"
                )
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "status": "ready",
                            "message": "trabajo de preparación existente, no necesita generación duplicada",
                            "already_prepared": True,
                            "prepare_info": prepare_info,
                        },
                    }
                )
            else:
                logger.info(
                    f"Simulación {simulation_id} no está preparado, iniciando tarea de preparación"
                )

        # Obtener información necesaria del proyecto
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify(
                {
                    "success": False,
                    "error": f"El proyecto no existe: {state.project_id}",
                }
            ), 404

        # Obtener requisito de simulación
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return jsonify(
                {
                    "success": False,
                    "error": "El proyecto no tiene descripción de requisito de simulación (simulation_requirement)",
                }
            ), 400

        # Obtener texto del documento
        document_text = ProjectManager.get_extracted_text(state.project_id) or ""

        entity_types_list = data.get("entity_types")
        use_llm_for_profiles = data.get("use_llm_for_profiles", True)
        parallel_profile_count = data.get("parallel_profile_count", 5)

        # ========== Obtener cantidad de entidades de forma síncrona（antes de iniciar tarea segundo plano） ==========
        # Así el frontend puede obtener el total esperado de Agents inmediatamente después de llamar a prepare
        try:
            logger.info(
                f"Obtener cantidad de entidades de forma síncrona: graph_id={state.graph_id}"
            )
            reader = ZepEntityReader()
            # Lectura rápida de entidades (sin información de bordes, solo contar cantidad)
            filtered_preview = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False,  # No obtener información de bordes, acelerar
            )
            # Guardar cantidad de entidades en estado (para que frontend obtenga inmediatamente)
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(
                f"Cantidad esperada de entidades: {filtered_preview.filtered_count}, Tipo: {filtered_preview.entity_types}"
            )
        except Exception as e:
            logger.warning(
                f"Obtener cantidad de entidades de forma síncrona falló（se reintentará en tarea segundo plano）: {e}"
            )
            # El fallo no afecta el flujo, tarea segundo plano volverá a obtener

        # Crear tarea asíncrona
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="simulation_prepare",
            metadata={"simulation_id": simulation_id, "project_id": state.project_id},
        )

        # Actualizar estado de simulación (incluye cantidad de entidades obtenida previamente)
        state.status = SimulationStatus.PREPARING
        manager._save_simulation_state(state)

        # Definir tarea segundo plano
        def run_prepare():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="Iniciando preparación del entorno de simulación...",
                )

                # Preparar simulación (con callback de progreso)
                # Almacenar detalles de fase de progreso
                stage_details = {}

                def progress_callback(stage, progress, message, **kwargs):
                    # Calcular progreso total
                    stage_weights = {
                        "reading": (0, 20),  # 0-20%
                        "generating_profiles": (20, 70),  # 20-70%
                        "generating_config": (70, 90),  # 70-90%
                        "copying_scripts": (90, 100),  # 90-100%
                    }

                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)

                    # Construir información detallada de progreso
                    stage_names = {
                        "reading": "lecturagraph_identidad",
                        "generating_profiles": "generaciónAgentperfil",
                        "generating_config": "generaciónSimulaciónconfiguración",
                        "copying_scripts": "script de preparación de simulación",
                    }

                    stage_index = (
                        list(stage_weights.keys()).index(stage) + 1
                        if stage in stage_weights
                        else 1
                    )
                    total_stages = len(stage_weights)

                    # Actualizar detalles de fase
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", ""),
                    }

                    # Construir información detallada de progreso
                    detail = stage_details[stage]
                    progress_detail_data = {
                        "current_stage": stage,
                        "current_stage_name": stage_names.get(stage, stage),
                        "stage_index": stage_index,
                        "total_stages": total_stages,
                        "stage_progress": progress,
                        "current_item": detail["current"],
                        "total_items": detail["total"],
                        "item_description": message,
                    }

                    # Construir mensaje conciso
                    if detail["total"] > 0:
                        detailed_message = (
                            f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: "
                            f"{detail['current']}/{detail['total']} - {message}"
                        )
                    else:
                        detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: {message}"

                    task_manager.update_task(
                        task_id,
                        progress=current_progress,
                        message=detailed_message,
                        progress_detail=progress_detail_data,
                    )

                result_state = manager.prepare_simulation(
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    defined_entity_types=entity_types_list,
                    use_llm_for_profiles=use_llm_for_profiles,
                    progress_callback=progress_callback,
                    parallel_profile_count=parallel_profile_count,
                )

                # tareacompletar
                task_manager.complete_task(
                    task_id, result=result_state.to_simple_dict()
                )

            except Exception as e:
                logger.error(f"Preparación de simulación falló: {str(e)}")
                task_manager.fail_task(task_id, str(e))

                # Actualizar estado de simulación a fallido
                state = manager.get_simulation(simulation_id)
                if state:
                    state.status = SimulationStatus.FAILED
                    state.error = str(e)
                    manager._save_simulation_state(state)

        # Iniciar hilo segundo plano
        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "task_id": task_id,
                    "status": "preparing",
                    "message": "La tarea de preparación ha iniciado, consulta el progreso en /api/simulation/prepare/status",
                    "already_prepared": False,
                    "expected_entities_count": state.entities_count,  # Cantidad total esperada de Agents
                    "entity_types": state.entity_types,  # lista de tipos de entidad
                },
            }
        )

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404

    except Exception as e:
        logger.error(f"Iniciar tarea de preparación falló: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/prepare/status", methods=["POST"])
def get_prepare_status():
    """
    Consultar progreso de tarea de preparación

    Soporta dos formas de consulta：
    1. Consultar progreso de tarea en curso mediante task_id
    2. Verificar si existe trabajo de preparación completado mediante simulation_id

    Solicitud (JSON):
        {
            "task_id": "task_xxxx",          // Opcional, task_id devuelto por prepare
            "simulation_id": "sim_xxxx"      // Opcional, ID de simulación (para verificar preparación completada)
        }

    Retorna:
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // Si existe trabajo de preparación completado
                "prepare_info": {...}            // Detalles cuando ya está preparado
            }
        }
    """
    from ..models.task import TaskManager

    try:
        data = request.get_json() or {}

        task_id = data.get("task_id")
        simulation_id = data.get("simulation_id")

        # Si se proporciona simulation_id, primero verificar si ya está preparado
        if simulation_id:
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            if is_prepared:
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "status": "ready",
                            "progress": 100,
                            "message": "trabajo de preparación existente completado",
                            "already_prepared": True,
                            "prepare_info": prepare_info,
                        },
                    }
                )

        # Si no hay task_id, retornar error
        if not task_id:
            if simulation_id:
                # Tiene simulation_id pero no está preparado completado
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "status": "not_started",
                            "progress": 0,
                            "message": "Aún no ha iniciado preparación, llama a /api/simulation/prepare para comenzar",
                            "already_prepared": False,
                        },
                    }
                )
            return jsonify(
                {
                    "success": False,
                    "error": "Por favor proporciona task_id o simulation_id",
                }
            ), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            # La tarea no existe, pero si hay simulation_id, verificar si ya está preparado
            if simulation_id:
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if is_prepared:
                    return jsonify(
                        {
                            "success": True,
                            "data": {
                                "simulation_id": simulation_id,
                                "task_id": task_id,
                                "status": "ready",
                                "progress": 100,
                                "message": "Tarea completada (trabajo de preparación ya existe)",
                                "already_prepared": True,
                                "prepare_info": prepare_info,
                            },
                        }
                    )

            return jsonify(
                {"success": False, "error": f"La tarea no existe: {task_id}"}
            ), 404

        task_dict = task.to_dict()
        task_dict["already_prepared"] = False

        return jsonify({"success": True, "data": task_dict})

    except Exception as e:
        logger.error(f"Consultar estado de tarea falló: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@simulation_bp.route("/<simulation_id>", methods=["GET"])
def get_simulation(simulation_id: str):
    """Obtener estado de simulación"""
    try:
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"La simulación no existe: {simulation_id}"}
            ), 404

        result = state.to_dict()

        # Si la simulación ya está lista, agregar instrucciones de ejecución
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = manager.get_run_instructions(simulation_id)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Obtener estado de simulación falló: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/list", methods=["GET"])
def list_simulations():
    """
    Listar todas las simulaciones

    Parámetros de consulta:
        project_id: Filtrar por ID de proyecto (opcional)
    """
    try:
        project_id = request.args.get("project_id")

        manager = SimulationManager()
        simulations = manager.list_simulations(project_id=project_id)

        return jsonify(
            {
                "success": True,
                "data": [s.to_dict() for s in simulations],
                "count": len(simulations),
            }
        )

    except Exception as e:
        logger.error(f"Listar simulaciones falló: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


def _get_report_id_for_simulation(simulation_id: str) -> str:
    """
    Obtener el último report_id correspondiente a la simulación

    Recorrer directorio de reports, encontrar reports que coincidan con simulation_id,
    Si hay múltiples, retornar el más reciente (ordenado por created_at)

    Args:
        simulation_id: SimulaciónID

    Returns:
        report_id o None
    """
    import json
    from datetime import datetime

    # Ruta del directorio de reports: backend/uploads/reports
    # __file__ es app/api/simulation.py, necesita subir dos niveles hasta backend/
    reports_dir = os.path.join(os.path.dirname(__file__), "../../uploads/reports")
    if not os.path.exists(reports_dir):
        return None

    matching_reports = []

    try:
        for report_folder in os.listdir(reports_dir):
            report_path = os.path.join(reports_dir, report_folder)
            if not os.path.isdir(report_path):
                continue

            meta_file = os.path.join(report_path, "meta.json")
            if not os.path.exists(meta_file):
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if meta.get("simulation_id") == simulation_id:
                    matching_reports.append(
                        {
                            "report_id": meta.get("report_id"),
                            "created_at": meta.get("created_at", ""),
                            "status": meta.get("status", ""),
                        }
                    )
            except Exception:
                continue

        if not matching_reports:
            return None

        # Ordenar por tiempo de creación descendente, retornar el más reciente
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")

    except Exception as e:
        logger.warning(f"Buscar simulation {simulation_id} del report falló: {e}")
        return None


@simulation_bp.route("/history", methods=["GET"])
def get_simulation_history():
    """
    Obtener lista de historial de simulación (con detalles del proyecto)

    Usado para mostrar historial en homepage, retorna lista de simulaciones con información rica como nombre y descripción del proyecto

    Parámetros de consulta:
        limit: Límite de cantidad a retornar (por defecto 20)

    Retorna:
        {
            "success": true,
            "data": [
                {
                    "simulation_id": "sim_xxxx",
                    "project_id": "proj_xxxx",
                    "project_name": "Analisis de opinion publica - Universidad de Wuhan",
                    "simulation_requirement": "Si la Universidad de Wuhan publica...",
                    "status": "completed",
                    "entities_count": 68,
                    "profiles_count": 68,
                    "entity_types": ["Student", "Professor", ...],
                    "created_at": "2024-12-10",
                    "updated_at": "2024-12-10",
                    "total_rounds": 120,
                    "current_round": 120,
                    "report_id": "report_xxxx",
                    "version": "v1.0.2"
                },
                ...
            ],
            "count": 7
        }
    """
    try:
        limit = request.args.get("limit", 20, type=int)

        manager = SimulationManager()
        simulations = manager.list_simulations()[:limit]

        # Enriquecer datos de simulación, solo leer de archivo Simulation
        enriched_simulations = []
        for sim in simulations:
            sim_dict = sim.to_dict()

            # Obtener información de configuración de simulación（leer simulation_requirement desde simulation_config.json）
            config = manager.get_simulation_config(sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get(
                    "simulation_requirement", ""
                )
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get(
                    "total_simulation_hours", 0
                )
                # Rondas recomendadas (valor de respaldo)
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0)
                    * 60
                    / max(time_config.get("minutes_per_round", 60), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0

            # Obtener estado de ejecución（leer rondas reales establecidas por el usuario desde run_state.json）
            run_state = SimulationRunner.get_run_state(sim.simulation_id)
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # Usar total_rounds establecido por el usuario, si no existe usar las recomendadas
                sim_dict["total_rounds"] = (
                    run_state.total_rounds
                    if run_state.total_rounds > 0
                    else recommended_rounds
                )
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds

            # Obtener lista de archivos asociados al proyecto (máximo 3)
            project = ProjectManager.get_project(sim.project_id)
            if project and hasattr(project, "files") and project.files:
                sim_dict["files"] = [
                    {"filename": f.get("filename", "archivo desconocido")}
                    for f in project.files[:3]
                ]
            else:
                sim_dict["files"] = []

            # Obtener report_id asociado（buscar el report mas reciente de esa simulacion）
            sim_dict["report_id"] = _get_report_id_for_simulation(sim.simulation_id)

            # Agregar número de versión
            sim_dict["version"] = "v1.0.2"

            # Formatear fecha
            try:
                created_date = sim_dict.get("created_at", "")[:10]
                sim_dict["created_date"] = created_date
            except:
                sim_dict["created_date"] = ""

            enriched_simulations.append(sim_dict)

        return jsonify(
            {
                "success": True,
                "data": enriched_simulations,
                "count": len(enriched_simulations),
            }
        )

    except Exception as e:
        logger.error(f"Obtener historial de Simulación falló: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/profiles", methods=["GET"])
def get_simulation_profiles(simulation_id: str):
    """
    Obtener Agent Profile de simulación

    Parámetros de consulta:
        platform: Tipo de plataforma（reddit/twitter，por defecto reddit)
    """
    try:
        platform = request.args.get("platform", "reddit")

        manager = SimulationManager()
        profiles = manager.get_profiles(simulation_id, platform=platform)

        return jsonify(
            {
                "success": True,
                "data": {
                    "platform": platform,
                    "count": len(profiles),
                    "profiles": profiles,
                },
            }
        )

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404

    except Exception as e:
        logger.error(f"Obtener Profile falló: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/profiles/realtime", methods=["GET"])
def get_simulation_profiles_realtime(simulation_id: str):
    """
    Obtener Agent Profile de simulacion en tiempo real (para ver progreso durante la generacion)

    Diferencias con la interfaz /profiles：
    - Lee archivo directamente, sin pasar por SimulationManager
    - Adecuado para ver en tiempo real durante la generacion
    - Devuelve metadatos adicionales (como tiempo de modificacion del archivo, Si esta generando, etc.)

    Parámetros de consulta:
        platform: Tipo de plataforma（reddit/twitter，por defecto reddit)

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "platform": "reddit",
                "count": 15,
                "total_expected": 93,  // Cantidad total esperada (si la hay)
                "is_generating": true,  // Si está generando
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "profiles": [...]
            }
        }
    """
    import json
    import csv
    from datetime import datetime

    try:
        platform = request.args.get("platform", "reddit")

        # Obtener directorio de simulación
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify(
                {"success": False, "error": f"La simulación no existe: {simulation_id}"}
            ), 404

        # Determinar ruta de archivo
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")

        # Verificar si archivo existe
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None

        if file_exists:
            # Obtener tiempo de modificación de archivo
            file_stat = os.stat(profiles_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            try:
                if platform == "reddit":
                    with open(profiles_file, "r", encoding="utf-8") as f:
                        profiles = json.load(f)
                else:
                    with open(profiles_file, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        profiles = list(reader)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    f"lectura profiles archivo falló (posiblemente escribiéndose): {e}"
                )
                profiles = []

        # Verificar Si esta generando (mediante verificacion de state.json)
        is_generating = False
        total_expected = None

        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    total_expected = state_data.get("entities_count")
            except Exception:
                pass

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "platform": platform,
                    "count": len(profiles),
                    "total_expected": total_expected,
                    "is_generating": is_generating,
                    "file_exists": file_exists,
                    "file_modified_at": file_modified_at,
                    "profiles": profiles,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error al obtener Profile en tiempo real{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config/realtime", methods=["GET"])
def get_simulation_config_realtime(simulation_id: str):
    """
    Obtener configuracion de simulacion en tiempo real (para ver progreso durante la generacion)

    Diferencias con la interfaz /config：
    - Lee archivo directamente, sin pasar por SimulationManager
    - Adecuado para ver en tiempo real durante la generacion
    - Devuelve metadatos adicionales (como tiempo de modificacion del archivo, Si esta generando, etc.)
    - incluso si la configuracion aun no esta completa puede devolver informacion parcial

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "is_generating": true,  // Si está generando
                "generation_stage": "generating_config",  // Fase actual de generacion
                "config": {...}  // contenido de configuracion (si existe)
            }
        }
    """
    import json
    from datetime import datetime

    try:
        # Obtener directorio de simulación
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify(
                {"success": False, "error": f"La simulación no existe: {simulation_id}"}
            ), 404

        # Ruta del archivo de configuracion
        config_file = os.path.join(sim_dir, "simulation_config.json")

        # Verificar si archivo existe
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None

        if file_exists:
            # Obtener tiempo de modificación de archivo
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(
                    f"lectura config archivo falló (posiblemente escribiéndose): {e}"
                )
                config = None

        # Verificar Si esta generando (mediante verificacion de state.json)
        is_generating = False
        generation_stage = None
        config_generated = False

        state_file = os.path.join(sim_dir, "state.json")
        if os.path.exists(state_file):
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    status = state_data.get("status", "")
                    is_generating = status == "preparing"
                    config_generated = state_data.get("config_generated", False)

                    # Determinar fase actual
                    if is_generating:
                        if state_data.get("profiles_generated", False):
                            generation_stage = "generating_config"
                        else:
                            generation_stage = "generating_profiles"
                    elif status == "ready":
                        generation_stage = "completed"
            except Exception:
                pass

        # Construir datos de retorno
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "generation_stage": generation_stage,
            "config_generated": config_generated,
            "config": config,
        }

        # Si configuración existe, extraer información de estadísticas clave
        if config:
            response_data["summary"] = {
                "total_agents": len(config.get("agent_configs", [])),
                "simulation_hours": config.get("time_config", {}).get(
                    "total_simulation_hours"
                ),
                "initial_posts_count": len(
                    config.get("event_config", {}).get("initial_posts", [])
                ),
                "hot_topics_count": len(
                    config.get("event_config", {}).get("hot_topics", [])
                ),
                "has_twitter_config": "twitter_config" in config,
                "has_reddit_config": "reddit_config" in config,
                "generated_at": config.get("generated_at"),
                "llm_model": config.get("llm_model"),
            }

        return jsonify({"success": True, "data": response_data})

    except Exception as e:
        logger.error(f"Obtener Config en tiempo real falló{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config", methods=["GET"])
def get_simulation_config(simulation_id: str):
    """
    Obtener configuración de simulación(configuracion completa generada inteligentemente por LLM)

    Retorna:
        - time_config: Configuración de tiempo（Duración de simulación, rondas, horas pico/valle）
        - agent_configs: Configuracion de actividad de cada Agent (Actividad, frecuencia de publicacion, posicion, etc.)
        - event_config: Configuracion de eventos (publicaciones iniciales, temas candentes)
        - platform_configs: Configuracion de plataforma
        - generation_reasoning: Explicacion del razonamiento de configuracion del LLM
    """
    try:
        manager = SimulationManager()
        config = manager.get_simulation_config(simulation_id)

        if not config:
            return jsonify(
                {
                    "success": False,
                    "error": "Configuracion de simulacion no existe, primero llame a /prepare",
                }
            ), 404

        return jsonify({"success": True, "data": config})

    except Exception as e:
        logger.error(f"Error al obtener configuracion{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config/download", methods=["GET"])
def download_simulation_config(simulation_id: str):
    """Descargar archivo de configuracion de Simulacion"""
    try:
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            return jsonify(
                {
                    "success": False,
                    "error": "Archivo de configuracion no existe, primero llame a /prepare",
                }
            ), 404

        return send_file(
            config_path, as_attachment=True, download_name="simulation_config.json"
        )

    except Exception as e:
        logger.error(f"Error al descargar configuracion{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/script/<script_name>/download", methods=["GET"])
def download_simulation_script(script_name: str):
    """
    Descargar script de ejecucion de Simulacion (script generico, ubicado en backend/scripts/)

    Valores opcionales de script_name：
        - run_twitter_simulation.py
        - run_reddit_simulation.py
        - run_parallel_simulation.py
        - action_logger.py
    """
    try:
        # Los scripts estan en el directorio backend/scripts/
        scripts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scripts")
        )

        # Validar nombre del script
        allowed_scripts = [
            "run_twitter_simulation.py",
            "run_reddit_simulation.py",
            "run_parallel_simulation.py",
            "action_logger.py",
        ]

        if script_name not in allowed_scripts:
            return jsonify(
                {
                    "success": False,
                    "error": f"Script desconocido: {script_name}, opciones: {allowed_scripts}",
                }
            ), 400

        script_path = os.path.join(scripts_dir, script_name)

        if not os.path.exists(script_path):
            return jsonify(
                {
                    "success": False,
                    "error": f"Archivo de script no existe: {script_name}",
                }
            ), 404

        return send_file(script_path, as_attachment=True, download_name=script_name)

    except Exception as e:
        logger.error(f"Error al descargar script{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de generacion de Perfiles (uso independiente) ==============


@simulation_bp.route("/generate-profiles", methods=["POST"])
def generate_profiles():
    """
    Generar directamente OASIS Agent Profile desde graph_id (sin crear Simulacion)

    Solicitud (JSON):
        {
            "graph_id": "mirofish_xxxx",     // Requerido
            "entity_types": ["Student"],      // Opcional
            "use_llm": true,                  // Opcional
            "platform": "reddit"              // Opcional
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")
        if not graph_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione graph_id"}
            ), 400

        entity_types = data.get("entity_types")
        use_llm = data.get("use_llm", True)
        platform = data.get("platform", "reddit")

        reader = ZepEntityReader()
        filtered = reader.filter_defined_entities(
            graph_id=graph_id, defined_entity_types=entity_types, enrich_with_edges=True
        )

        if filtered.filtered_count == 0:
            return jsonify(
                {
                    "success": False,
                    "error": "No se encontraron entidades que coincidan con los criterios",
                }
            ), 400

        generator = OasisProfileGenerator()
        profiles = generator.generate_profiles_from_entities(
            entities=filtered.entities, use_llm=use_llm
        )

        if platform == "reddit":
            profiles_data = [p.to_reddit_format() for p in profiles]
        elif platform == "twitter":
            profiles_data = [p.to_twitter_format() for p in profiles]
        else:
            profiles_data = [p.to_dict() for p in profiles]

        return jsonify(
            {
                "success": True,
                "data": {
                    "platform": platform,
                    "entity_types": list(filtered.entity_types),
                    "count": len(profiles_data),
                    "profiles": profiles_data,
                },
            }
        )

    except Exception as e:
        logger.error(f"generaciónProfilefalló{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de control de ejecucion de Simulacion ==============


@simulation_bp.route("/start", methods=["POST"])
def start_simulation():
    """
    Iniciar ejecucion de Simulacion

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",          // Requerido, ID de Simulacion
            "platform": "parallel",                // Opcional: twitter / reddit / parallel (por defecto)
            "max_rounds": 100,                     // Opcional: Maximo de rondas de Simulacion, para truncar simulaciones muy largas
            "enable_graph_memory_update": false,   // Opcional: Si actualizar dinamicamente la memoria del grafo Zep con actividades de los Agents
            "force": false                         // Opcional: Reinicio forzado (detiene la Simulacion en ejecucion y limpia los logs)
        }

    Acerca del parametros force：
        - Si esta habilitado, si la Simulacion esta en ejecucion o ya completada, primero se detendra y limpiara los logs
        - El contenido a limpiar incluye: run_state.json, actions.jsonl, simulation.log, etc.
        - No limpiara archivos de configuracion (simulation_config.json) ni archivos de perfiles
        - Adecuado para escenarios donde se necesita volver a ejecutar la Simulacion

    Acerca de enable_graph_memory_update：
        - Si esta habilitado, todas las actividades de los Agents en la Simulacion (publicar, comentar, dar like, etc.) se actualizaran en tiempo real en el grafo Zep
        - Esto permite que el grafo "recuerde" el proceso de Simulacion, para analisis posterior o conversacion con IA
        - Requiere que el proyecto asociado a la Simulacion tenga un graph_id valido
        - Utiliza mecanismo de actualizacion por lotes para reducir llamadas API

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "process_pid": 12345,
                "twitter_running": true,
                "reddit_running": true,
                "started_at": "2025-12-01T10:00:00",
                "graph_memory_update_enabled": true,  // Si se habilito la actualizacion de memoria del grafo
                "force_restarted": true               // Si es un reinicio forzado
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        platform = data.get("platform", "parallel")
        max_rounds = data.get("max_rounds")  # Opcional: Maximo de rondas de Simulacion
        enable_graph_memory_update = data.get(
            "enable_graph_memory_update", False
        )  # Opcional: Si habilitar actualizacion de memoria del grafo
        force = data.get("force", False)  # Opcional: Reinicio forzado

        # Validar parametros max_rounds
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return jsonify(
                        {"success": False, "error": "max_rounds debe ser un entero positivo"}
                    ), 400
            except (ValueError, TypeError):
                return jsonify(
                    {"success": False, "error": "max_rounds debe ser un entero valido"}
                ), 400

        if platform not in ["twitter", "reddit", "parallel"]:
            return jsonify(
                {
                    "success": False,
                    "error": f"Tipo de plataforma invalido: {platform}，opcional: twitter/reddit/parallel",
                }
            ), 400

        # Verificar si la simulacion ya esta preparada
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"La simulación no existe: {simulation_id}"}
            ), 404

        force_restarted = False

        # Manejo inteligente del estado: si el trabajo de preparacion esta completado, permite reiniciar
        if state.status != SimulationStatus.READY:
            # Verificar si el trabajo de preparacion ya esta completado
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)

            if is_prepared:
                # Trabajo de preparacion completado, verificar si hay procesos en ejecucion
                if state.status == SimulationStatus.RUNNING:
                    # Verificar si el proceso de simulacion realmente esta en ejecucion
                    run_state = SimulationRunner.get_run_state(simulation_id)
                    if run_state and run_state.runner_status.value == "running":
                        # El proceso realmente esta en ejecucion
                        if force:
                            # Modo forzado: detener simulación en ejecución
                            logger.info(
                                f"Modo forzado: deteniendo simulación en ejecución {simulation_id}"
                            )
                            try:
                                SimulationRunner.stop_simulation(simulation_id)
                            except Exception as e:
                                logger.warning(
                                    f"Advertencia al detener simulación: {str(e)}"
                                )
                        else:
                            return jsonify(
                                {
                                    "success": False,
                                    "error": "La simulación está en ejecución. Primero llame a /stop para detener, o use force=true para reiniciar forzadamente",
                                }
                            ), 400

                # Si es modo forzado, limpiar logs de ejecución
                if force:
                    logger.info(
                        f"Modo forzado: limpiando logs de simulación {simulation_id}"
                    )
                    cleanup_result = SimulationRunner.cleanup_simulation_logs(
                        simulation_id
                    )
                    if not cleanup_result.get("success"):
                        logger.warning(
                            f"Limpieza de logs generó advertencia: {cleanup_result.get('errors')}"
                        )
                    force_restarted = True

                # Proceso no existe o ya terminó, resetear estado a ready
                logger.info(
                    f"Simulación {simulation_id} preparación completada, reseteando estado a ready (estado original: {state.status.value})"
                )
                state.status = SimulationStatus.READY
                manager._save_simulation_state(state)
            else:
                # Preparación no completada
                return jsonify(
                    {
                        "success": False,
                        "error": f"Simulación no está lista, estado actual: {state.status.value}. Primero llame a /prepare",
                    }
                ), 400

        # Obtener graph_id (para actualización de memoria del grafo)
        graph_id = None
        if enable_graph_memory_update:
            # Desde estado de simulación o proyecto obtener graph_id
            graph_id = state.graph_id
            if not graph_id:
                # Intentar obtener del proyecto
                project = ProjectManager.get_project(state.project_id)
                if project:
                    graph_id = project.graph_id

            if not graph_id:
                return jsonify(
                    {
                        "success": False,
                        "error": "Para habilitar actualización de memoria del grafo se requiere graph_id válido. Asegúrese de que el proyecto tenga grafo construido",
                    }
                ), 400

            logger.info(
                f"Habilitando actualización de memoria del grafo: simulation_id={simulation_id}, graph_id={graph_id}"
            )

        # iniciarSimulación
        run_state = SimulationRunner.start_simulation(
            simulation_id=simulation_id,
            platform=platform,
            max_rounds=max_rounds,
            enable_graph_memory_update=enable_graph_memory_update,
            graph_id=graph_id,
        )

        # Actualizar estado de Simulacion
        state.status = SimulationStatus.RUNNING
        manager._save_simulation_state(state)

        response_data = run_state.to_dict()
        if max_rounds:
            response_data["max_rounds_applied"] = max_rounds
        response_data["graph_memory_update_enabled"] = enable_graph_memory_update
        response_data["force_restarted"] = force_restarted
        if enable_graph_memory_update:
            response_data["graph_id"] = graph_id

        return jsonify({"success": True, "data": response_data})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"iniciarSimulaciónfalló{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/stop", methods=["POST"])
def stop_simulation():
    """
    Detener Simulacion

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx"  // Requerido, ID de Simulacion
        }

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "stopped",
                "completed_at": "2025-12-01T12:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        run_state = SimulationRunner.stop_simulation(simulation_id)

        # Actualizar estado de Simulacion
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.PAUSED
            manager._save_simulation_state(state)

        return jsonify({"success": True, "data": run_state.to_dict()})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"Error al detener Simulacion{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de monitoreo de estado en tiempo real ==============


@simulation_bp.route("/<simulation_id>/run-status", methods=["GET"])
def get_run_status(simulation_id: str):
    """
    Obtener estado de ejecucion de Simulacion en tiempo real (para polling del frontend)

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                "total_rounds": 144,
                "progress_percent": 3.5,
                "simulated_hours": 2,
                "total_simulation_hours": 72,
                "twitter_running": true,
                "reddit_running": true,
                "twitter_actions_count": 150,
                "reddit_actions_count": 200,
                "total_actions_count": 350,
                "started_at": "2025-12-01T10:00:00",
                "updated_at": "2025-12-01T10:30:00"
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)

        if not run_state:
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "runner_status": "idle",
                        "current_round": 0,
                        "total_rounds": 0,
                        "progress_percent": 0,
                        "twitter_actions_count": 0,
                        "reddit_actions_count": 0,
                        "total_actions_count": 0,
                    },
                }
            )

        return jsonify({"success": True, "data": run_state.to_dict()})

    except Exception as e:
        logger.error(f"Obtener estado de ejecuciónfalló{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/run-status/detail", methods=["GET"])
def get_run_status_detail(simulation_id: str):
    """
    Obtener estado detallado de ejecucion de Simulacion (contiene todas las acciones)

    Para mostrar dinamicas en tiempo real en el frontend

    Parámetros de consulta:
        platform: Filtrar plataforma (twitter/reddit, opcional)

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "current_round": 5,
                ...
                "all_actions": [
                    {
                        "round_num": 5,
                        "timestamp": "2025-12-01T10:30:00",
                        "platform": "twitter",
                        "agent_id": 3,
                        "agent_name": "Agent Name",
                        "action_type": "CREATE_POST",
                        "action_args": {"content": "..."},
                        "result": null,
                        "success": true
                    },
                    ...
                ],
                "twitter_actions": [...],  # Todas las acciones de la plataforma Twitter
                "reddit_actions": [...]    # Todas las acciones de la plataforma Reddit
            }
        }
    """
    try:
        run_state = SimulationRunner.get_run_state(simulation_id)
        platform_filter = request.args.get("platform")

        if not run_state:
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "runner_status": "idle",
                        "all_actions": [],
                        "twitter_actions": [],
                        "reddit_actions": [],
                    },
                }
            )

        # Obtener la lista completa de acciones
        all_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id, platform=platform_filter
        )

        # Obtener acciones por plataforma
        twitter_actions = (
            SimulationRunner.get_all_actions(
                simulation_id=simulation_id, platform="twitter"
            )
            if not platform_filter or platform_filter == "twitter"
            else []
        )

        reddit_actions = (
            SimulationRunner.get_all_actions(
                simulation_id=simulation_id, platform="reddit"
            )
            if not platform_filter or platform_filter == "reddit"
            else []
        )

        # Obtener acciones de la ronda actual (recent_actions solo muestra la ultima ronda)
        current_round = run_state.current_round
        recent_actions = (
            SimulationRunner.get_all_actions(
                simulation_id=simulation_id,
                platform=platform_filter,
                round_num=current_round,
            )
            if current_round > 0
            else []
        )

        # Obtener informacion basica del estado
        result = run_state.to_dict()
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        # recent_actions solo muestra contenido de la ultima ronda de ambas plataformas
        result["recent_actions"] = [a.to_dict() for a in recent_actions]

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"Error al obtener estado detallado{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/actions", methods=["GET"])
def get_simulation_actions(simulation_id: str):
    """
    Obtener historial de acciones de Agents en Simulacion

    Parámetros de consulta:
        limit: Cantidad a devolver (por defecto 100)
        offset: Desplazamiento (por defecto 0)
        platform: Filtrar plataforma (twitter/reddit)
        agent_id: Filtrar Agent ID
        round_num: Filtrar ronda

    Retorna:
        {
            "success": true,
            "data": {
                "count": 100,
                "actions": [...]
            }
        }
    """
    try:
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)
        platform = request.args.get("platform")
        agent_id = request.args.get("agent_id", type=int)
        round_num = request.args.get("round_num", type=int)

        actions = SimulationRunner.get_actions(
            simulation_id=simulation_id,
            limit=limit,
            offset=offset,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num,
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "count": len(actions),
                    "actions": [a.to_dict() for a in actions],
                },
            }
        )

    except Exception as e:
        logger.error(f"Error al obtener historial de acciones{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/timeline", methods=["GET"])
def get_simulation_timeline(simulation_id: str):
    """
    Obtener linea de tiempo de Simulacion (resumen por rondas)

    Para mostrar barra de progreso y vista de linea de tiempo en frontend

    Parámetros de consulta:
        start_round: Ronda inicial (por defecto 0)
        end_round: Ronda final (por defecto todas)

    Devolver informacion resumida de cada ronda
    """
    try:
        start_round = request.args.get("start_round", 0, type=int)
        end_round = request.args.get("end_round", type=int)

        timeline = SimulationRunner.get_timeline(
            simulation_id=simulation_id, start_round=start_round, end_round=end_round
        )

        return jsonify(
            {
                "success": True,
                "data": {"rounds_count": len(timeline), "timeline": timeline},
            }
        )

    except Exception as e:
        logger.error(f"Error al obtener linea de tiempo{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/agent-stats", methods=["GET"])
def get_agent_stats(simulation_id: str):
    """
    Obtener estadisticas de cada Agent

    Para mostrar ranking de actividad de Agents y distribucion de acciones en frontend
    """
    try:
        stats = SimulationRunner.get_agent_stats(simulation_id)

        return jsonify(
            {"success": True, "data": {"agents_count": len(stats), "stats": stats}}
        )

    except Exception as e:
        logger.error(f"Error al obtener estadisticas de Agents{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de consulta de base de datos ==============


@simulation_bp.route("/<simulation_id>/posts", methods=["GET"])
def get_simulation_posts(simulation_id: str):
    """
    Obtener publicaciones de Simulacion

    Parámetros de consulta:
        platform: Tipo de plataforma（twitter/reddit）
        limit: Cantidad a devolver (por defecto 50)
        offset: Desplazamiento

    Devolver lista de publicaciones (leer desde base de datos SQLite)
    """
    try:
        platform = request.args.get("platform", "reddit")
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)

        sim_dir = os.path.join(
            os.path.dirname(__file__), f"../../uploads/simulations/{simulation_id}"
        )

        db_file = f"{platform}_simulation.db"
        db_path = os.path.join(sim_dir, db_file)

        if not os.path.exists(db_path):
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "platform": platform,
                        "count": 0,
                        "posts": [],
                        "message": "Base de datos no existe, la Simulacion quizas aun no se ha ejecutado",
                    },
                }
            )

        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT * FROM post 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """,
                (limit, offset),
            )

            posts = [dict(row) for row in cursor.fetchall()]

            cursor.execute("SELECT COUNT(*) FROM post")
            total = cursor.fetchone()[0]

        except sqlite3.OperationalError:
            posts = []
            total = 0

        conn.close()

        return jsonify(
            {
                "success": True,
                "data": {
                    "platform": platform,
                    "total": total,
                    "count": len(posts),
                    "posts": posts,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error al obtener publicaciones{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/comments", methods=["GET"])
def get_simulation_comments(simulation_id: str):
    """
    Obtener comentarios de Simulacion (solo Reddit)

    Parámetros de consulta:
        post_id: Filtrar ID de publicacion (opcional)
        limit: Cantidad a devolver
        offset: Desplazamiento
    """
    try:
        post_id = request.args.get("post_id")
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)

        sim_dir = os.path.join(
            os.path.dirname(__file__), f"../../uploads/simulations/{simulation_id}"
        )

        db_path = os.path.join(sim_dir, "reddit_simulation.db")

        if not os.path.exists(db_path):
            return jsonify({"success": True, "data": {"count": 0, "comments": []}})

        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            if post_id:
                cursor.execute(
                    """
                    SELECT * FROM comment 
                    WHERE post_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """,
                    (post_id, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM comment 
                    ORDER BY created_at DESC 
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )

            comments = [dict(row) for row in cursor.fetchall()]

        except sqlite3.OperationalError:
            comments = []

        conn.close()

        return jsonify(
            {"success": True, "data": {"count": len(comments), "comments": comments}}
        )

    except Exception as e:
        logger.error(f"Error al obtener comentarios{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de Interview ==============


@simulation_bp.route("/interview", methods=["POST"])
def interview_agent():
    """
    Entrevistar un Agent individual

    Nota: Esta funcion requiere que el entorno de Simulacion este en ejecucion

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",       // Requerido, ID de Simulacion
            "agent_id": 0,                     // Requerido, Agent ID
            "prompt": "Que opinas sobre esto?",  // Requerido, pregunta de entrevista
            "platform": "twitter",             // Opcional, especificar plataforma (twitter/reddit)
                                               // Si no se especifica: Simulacion de doble plataforma entrevista ambas plataformas
            "timeout": 60                      // Opcional, tiempo de espera en segundos, por defecto 60
        }

    Retorno (sin especificar platform, modo dual)：
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "Que opinas sobre esto?",
                "result": {
                    "agent_id": 0,
                    "prompt": "...",
                    "platforms": {
                        "twitter": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit": {"agent_id": 0, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }

    Retorno (especificando platform)：
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "Que opinas sobre esto?",
                "result": {
                    "agent_id": 0,
                    "response": "Creo que...",
                    "platform": "twitter",
                    "timestamp": "2025-12-08T10:00:00"
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        agent_id = data.get("agent_id")
        prompt = data.get("prompt")
        platform = data.get("platform")  # Opcional: twitter/reddit/None
        timeout = data.get("timeout", 60)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        if agent_id is None:
            return jsonify(
                {"success": False, "error": "Por favor proporcione agent_id"}
            ), 400

        if not prompt:
            return jsonify(
                {
                    "success": False,
                    "error": "Por favor proporcione prompt (pregunta de entrevista)",
                }
            ), 400

        # Verificar parámetro platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {
                    "success": False,
                    "error": "El parámetro platform solo puede ser 'twitter' o 'reddit'",
                }
            ), 400

        # Verificar estado del entorno
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify(
                {
                    "success": False,
                    "error": "Entorno de simulación no está ejecutándose o está cerrado. Asegúrese de que la simulación haya completado y esté en modo de espera de comandos.",
                }
            ), 400

        # Optimizar prompt, agregar prefijo para evitar que Agent llame herramientas
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_agent(
            simulation_id=simulation_id,
            agent_id=agent_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout,
        )

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except TimeoutError as e:
        return jsonify(
            {"success": False, "error": f"Esperando respuesta de Interview agotado: {str(e)}"}
        ), 504

    except Exception as e:
        logger.error(f"Error en Interview{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/batch", methods=["POST"])
def interview_agents_batch():
    """
    Entrevistar multiples Agents en lote

    Nota: Esta funcion requiere que el entorno de Simulacion este en ejecucion

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",       // Requerido, ID de Simulacion
            "interviews": [                    // Requerido, lista de entrevistas
                {
                    "agent_id": 0,
                    "prompt": "Que opinas de A?",
                    "platform": "twitter"      // Opcional, especificar plataforma de entrevista de ese Agent
                },
                {
                    "agent_id": 1,
                    "prompt": "Que opinas de B?"  // Si no se especifica platform usa el valor por defecto
                }
            ],
            "platform": "reddit",              // Opcional, plataforma por defecto (sobrescrita por cada item)
                                               // Si no se especifica: Simulacion de doble plataforma entrevista ambas plataformas para cada Agent
            "timeout": 120                     // Opcional, tiempo de espera en segundos, por defecto 120
        }

    Retorna:
        {
            "success": true,
            "data": {
                "interviews_count": 2,
                "result": {
                    "interviews_count": 4,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        "twitter_1": {"agent_id": 1, "response": "...", "platform": "twitter"},
                        "reddit_1": {"agent_id": 1, "response": "...", "platform": "reddit"}
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        interviews = data.get("interviews")
        platform = data.get("platform")  # Opcional: twitter/reddit/None
        timeout = data.get("timeout", 120)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        if not interviews or not isinstance(interviews, list):
            return jsonify(
                {
                    "success": False,
                    "error": "Por favor proporcione interviews (lista de entrevistas)",
                }
            ), 400

        # Verificar parámetro platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {
                    "success": False,
                    "error": "El parámetro platform solo puede ser 'twitter' o 'reddit'",
                }
            ), 400

        # Verificar cada elemento de entrevista
        for i, interview in enumerate(interviews):
            if "agent_id" not in interview:
                return jsonify(
                    {
                        "success": False,
                        "error": f"El elemento {i + 1} de la lista de entrevistas carece de agent_id",
                    }
                ), 400
            if "prompt" not in interview:
                return jsonify(
                    {
                        "success": False,
                        "error": f"El elemento {i + 1} de la lista de entrevistas carece de prompt",
                    }
                ), 400
            # Verificar platform de cada elemento (si existe)
            item_platform = interview.get("platform")
            if item_platform and item_platform not in ("twitter", "reddit"):
                return jsonify(
                    {
                        "success": False,
                        "error": f"El elemento {i + 1} de la lista de entrevistas solo puede tener platform 'twitter' o 'reddit'",
                    }
                ), 400

        # Verificar estado del entorno
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify(
                {
                    "success": False,
                    "error": "El entorno de Simulacion no esta en ejecucion o esta cerrado. Asegurese de que la Simulacion haya completado y entrado en modo de espera de comandos.",
                }
            ), 400

        # Optimizar cada prompt de entrevista, agregar prefijo para evitar que el Agent llame herramientas
        optimized_interviews = []
        for interview in interviews:
            optimized_interview = interview.copy()
            optimized_interview["prompt"] = optimize_interview_prompt(
                interview.get("prompt", "")
            )
            optimized_interviews.append(optimized_interview)

        result = SimulationRunner.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=optimized_interviews,
            platform=platform,
            timeout=timeout,
        )

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except TimeoutError as e:
        return jsonify(
            {"success": False, "error": f"Esperando respuesta de Interview por lotes agotado: {str(e)}"}
        ), 504

    except Exception as e:
        logger.error(f"Error en Interview por lotes{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/all", methods=["POST"])
def interview_all_agents():
    """
    Entrevista global - Usar la misma pregunta para entrevistar a todos los Agents

    Nota: Esta funcion requiere que el entorno de Simulacion este en ejecucion

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",            // Requerido, ID de Simulacion
            "prompt": "Que opinas en general sobre esto?",  // Requerido, pregunta de entrevista (todos los Agents usan la misma pregunta)
            "platform": "reddit",                   // Opcional, especificar plataforma (twitter/reddit)
                                                    // Si no se especifica: Simulacion de doble plataforma entrevista ambas plataformas para cada Agent
            "timeout": 180                          // Opcional, tiempo de espera en segundos, por defecto 180
        }

    Retorna:
        {
            "success": true,
            "data": {
                "interviews_count": 50,
                "result": {
                    "interviews_count": 100,
                    "results": {
                        "twitter_0": {"agent_id": 0, "response": "...", "platform": "twitter"},
                        "reddit_0": {"agent_id": 0, "response": "...", "platform": "reddit"},
                        ...
                    }
                },
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        prompt = data.get("prompt")
        platform = data.get("platform")  # Opcional: twitter/reddit/None
        timeout = data.get("timeout", 180)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        if not prompt:
            return jsonify(
                {"success": False, "error": "Por favor proporcione prompt (pregunta de entrevista)"}
            ), 400

        # Validar parametros platform
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {"success": False, "error": "El parametro platform solo puede ser twitter o reddit"}
            ), 400

        # Verificar estado del entorno
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify(
                {
                    "success": False,
                    "error": "El entorno de Simulacion no esta en ejecucion o esta cerrado. Asegurese de que la Simulacion haya completado y entrado en modo de espera de comandos.",
                }
            ), 400

        # Optimizar prompt, agregar prefijo para evitar que el Agent llame herramientas
        optimized_prompt = optimize_interview_prompt(prompt)

        result = SimulationRunner.interview_all_agents(
            simulation_id=simulation_id,
            prompt=optimized_prompt,
            platform=platform,
            timeout=timeout,
        )

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except TimeoutError as e:
        return jsonify(
            {"success": False, "error": f"Esperando respuesta de Interview global agotado: {str(e)}"}
        ), 504

    except Exception as e:
        logger.error(f"Error en Interview global{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/history", methods=["POST"])
def get_interview_history():
    """
    Obtener registro de historial de Interview

    Leer todos los registros de Interview desde la base de datos de Simulacion

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",  // Requerido, ID de Simulacion
            "platform": "reddit",          // Opcional, Tipo de plataforma (reddit/twitter)
                                           // Si no se especifica devuelve historial de ambas plataformas
            "agent_id": 0,                 // Opcional，solo obtener historial de entrevistas de ese Agent
            "limit": 100                   // Opcional, cantidad a devolver, por defecto 100
        }

    Retorna:
        {
            "success": true,
            "data": {
                "count": 10,
                "history": [
                    {
                        "agent_id": 0,
                        "response": "Creo que...",
                        "prompt": "Que opinas sobre esto?",
                        "timestamp": "2025-12-08T10:00:00",
                        "platform": "reddit"
                    },
                    ...
                ]
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        platform = data.get("platform")  # Si no se especifica devuelve historial de ambas plataformas
        agent_id = data.get("agent_id")
        limit = data.get("limit", 100)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        history = SimulationRunner.get_interview_history(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            limit=limit,
        )

        return jsonify(
            {"success": True, "data": {"count": len(history), "history": history}}
        )

    except Exception as e:
        logger.error(f"Error al obtener historial de Interview{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/env-status", methods=["POST"])
def get_env_status():
    """
    Obtener estado del entorno de Simulacion

    Verificar si el entorno de simulacion esta vivo (puede recibir comandos de Interview)

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx"  // Requerido, ID de Simulacion
        }

    Retorna:
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "env_alive": true,
                "twitter_available": true,
                "reddit_available": true,
                "message": "El entorno esta en ejecucion, puede recibir comandos de Interview"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        env_alive = SimulationRunner.check_env_alive(simulation_id)

        # Obtener informacion de estado mas detallada
        env_status = SimulationRunner.get_env_status_detail(simulation_id)

        if env_alive:
            message = "El entorno esta en ejecucion, puede recibir comandos de Interview"
        else:
            message = "El entorno no esta en ejecucion o esta cerrado"

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "env_alive": env_alive,
                    "twitter_available": env_status.get("twitter_available", False),
                    "reddit_available": env_status.get("reddit_available", False),
                    "message": message,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error al obtener estado del entorno{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/close-env", methods=["POST"])
def close_simulation_env():
    """
    Cerrar entorno de Simulacion

    Enviar comando de cierre a Simulacion para que salga elegantemente del modo de espera de comandos.

    Nota: Esto es diferente de la interfaz /stop, /stop fuerza la terminacion del proceso,
    esta interfaz hace que Simulacion cierre elegantemente el entorno y salga.

    Solicitud (JSON):
        {
            "simulation_id": "sim_xxxx",  // Requerido, ID de Simulacion
            "timeout": 30                  // Opcional, tiempo de espera en segundos, por defecto 30
        }

    Retorna:
        {
            "success": true,
            "data": {
                "message": "Comando de cierre de entorno enviado",
                "result": {...},
                "timestamp": "2025-12-08T10:00:01"
            }
        }
    """
    try:
        data = request.get_json() or {}

        simulation_id = data.get("simulation_id")
        timeout = data.get("timeout", 30)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        result = SimulationRunner.close_simulation_env(
            simulation_id=simulation_id, timeout=timeout
        )

        # Actualizar estado de Simulacion
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            manager._save_simulation_state(state)

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"Error al cerrar entorno{str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500
