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
                {"success": False, "error": f"Simulación不存在: {simulation_id}"}
            ), 404

        # 检查是否强制重新generación
        force_regenerate = data.get("force_regenerate", False)
        logger.info(
            f"开始处理 /prepare 请求: simulation_id={simulation_id}, force_regenerate={force_regenerate}"
        )

        # 检查是否已经prepararcompletar（避免重复generación）
        if not force_regenerate:
            logger.debug(f"检查Simulación {simulation_id} 是否已prepararcompletar...")
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            logger.debug(
                f"检查结果: is_prepared={is_prepared}, prepare_info={prepare_info}"
            )
            if is_prepared:
                logger.info(
                    f"Simulación {simulation_id} 已prepararcompletar，跳过重复generación"
                )
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "status": "ready",
                            "message": "existentecompletar的preparar工作，无需重复generación",
                            "already_prepared": True,
                            "prepare_info": prepare_info,
                        },
                    }
                )
            else:
                logger.info(
                    f"Simulación {simulation_id} 未prepararcompletar，将iniciarpreparartarea"
                )

        # 从proyecto获取必要信息
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify(
                {"success": False, "error": f"proyecto不存在: {state.project_id}"}
            ), 404

        # 获取Simulación需求
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return jsonify(
                {
                    "success": False,
                    "error": "proyecto缺少Simulación需求描述 (simulation_requirement)",
                }
            ), 400

        # 获取文档文本
        document_text = ProjectManager.get_extracted_text(state.project_id) or ""

        entity_types_list = data.get("entity_types")
        use_llm_for_profiles = data.get("use_llm_for_profiles", True)
        parallel_profile_count = data.get("parallel_profile_count", 5)

        # ========== 同步获取entidad数量（在后台tareainiciar前） ==========
        # 这样前端在调用prepare后立即就能获取到预期Agent总数
        try:
            logger.info(f"同步获取entidad数量: graph_id={state.graph_id}")
            reader = ZepEntityReader()
            # 快速lecturaentidad（不需要边信息，只统计数量）
            filtered_preview = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False,  # 不获取边信息，加快速度
            )
            # 保存entidad数量到estado（供前端立即获取）
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(
                f"预期entidad数量: {filtered_preview.filtered_count}, 类型: {filtered_preview.entity_types}"
            )
        except Exception as e:
            logger.warning(f"同步获取entidad数量失败（将在后台tarea中重试）: {e}")
            # 失败不影响后续流程，后台tarea会重新获取

        # 创建异步tarea
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="simulation_prepare",
            metadata={"simulation_id": simulation_id, "project_id": state.project_id},
        )

        # 更新Simulaciónestado（包含预先获取的entidad数量）
        state.status = SimulationStatus.PREPARING
        manager._save_simulation_state(state)

        # 定义后台tarea
        def run_prepare():
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message="开始prepararSimulación环境...",
                )

                # prepararSimulación（带progreso回调）
                # 存储faseprogreso详情
                stage_details = {}

                def progress_callback(stage, progress, message, **kwargs):
                    # 计算总progreso
                    stage_weights = {
                        "reading": (0, 20),  # 0-20%
                        "generating_profiles": (20, 70),  # 20-70%
                        "generating_config": (70, 90),  # 70-90%
                        "copying_scripts": (90, 100),  # 90-100%
                    }

                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)

                    # 构建详细progreso信息
                    stage_names = {
                        "reading": "lecturagraph_identidad",
                        "generating_profiles": "generaciónAgentperfil",
                        "generating_config": "generaciónSimulaciónconfiguración",
                        "copying_scripts": "prepararSimulación脚本",
                    }

                    stage_index = (
                        list(stage_weights.keys()).index(stage) + 1
                        if stage in stage_weights
                        else 1
                    )
                    total_stages = len(stage_weights)

                    # 更新fase详情
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", ""),
                    }

                    # 构建详细progreso信息
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

                    # 构建简洁消息
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
                logger.error(f"prepararSimulación失败: {str(e)}")
                task_manager.fail_task(task_id, str(e))

                # 更新Simulaciónestado为失败
                state = manager.get_simulation(simulation_id)
                if state:
                    state.status = SimulationStatus.FAILED
                    state.error = str(e)
                    manager._save_simulation_state(state)

        # iniciar后台线程
        thread = threading.Thread(target=run_prepare, daemon=True)
        thread.start()

        return jsonify(
            {
                "success": True,
                "data": {
                    "simulation_id": simulation_id,
                    "task_id": task_id,
                    "status": "preparing",
                    "message": "preparartarea已iniciar，请通过 /api/simulation/prepare/status consultaprogreso",
                    "already_prepared": False,
                    "expected_entities_count": state.entities_count,  # 预期的Agent总数
                    "entity_types": state.entity_types,  # entidad类型列表
                },
            }
        )

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404

    except Exception as e:
        logger.error(f"iniciarpreparartarea失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/prepare/status", methods=["POST"])
def get_prepare_status():
    """
    consultapreparartareaprogreso

    支持两种consulta方式：
    1. 通过task_idconsulta正在进行的tareaprogreso
    2. 通过simulation_id检查是否existentecompletar的preparar工作

    请求（JSON）：
        {
            "task_id": "task_xxxx",          // 可选，prepare返回的task_id
            "simulation_id": "sim_xxxx"      // 可选，SimulaciónID（用于检查已completar的preparar）
        }

    返回：
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // 是否existentecompletar的preparar
                "prepare_info": {...}            // 已prepararcompletar时的详细信息
            }
        }
    """
    from ..models.task import TaskManager

    try:
        data = request.get_json() or {}

        task_id = data.get("task_id")
        simulation_id = data.get("simulation_id")

        # 如果提供了simulation_id，先检查是否已prepararcompletar
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
                            "message": "existentecompletar的preparar工作",
                            "already_prepared": True,
                            "prepare_info": prepare_info,
                        },
                    }
                )

        # 如果没有task_id，返回错误
        if not task_id:
            if simulation_id:
                # 有simulation_id但未prepararcompletar
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "status": "not_started",
                            "progress": 0,
                            "message": "尚未开始preparar，请调用 /api/simulation/prepare 开始",
                            "already_prepared": False,
                        },
                    }
                )
            return jsonify(
                {"success": False, "error": "请提供 task_id 或 simulation_id"}
            ), 400

        task_manager = TaskManager()
        task = task_manager.get_task(task_id)

        if not task:
            # tarea不存在，但如果有simulation_id，检查是否已prepararcompletar
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
                                "message": "tarea已completar（preparar工作已存在）",
                                "already_prepared": True,
                                "prepare_info": prepare_info,
                            },
                        }
                    )

            return jsonify({"success": False, "error": f"tarea不存在: {task_id}"}), 404

        task_dict = task.to_dict()
        task_dict["already_prepared"] = False

        return jsonify({"success": True, "data": task_dict})

    except Exception as e:
        logger.error(f"consultatareaestado失败: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@simulation_bp.route("/<simulation_id>", methods=["GET"])
def get_simulation(simulation_id: str):
    """获取Simulaciónestado"""
    try:
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"Simulación不存在: {simulation_id}"}
            ), 404

        result = state.to_dict()

        # 如果Simulación已preparar好，附加运行说明
        if state.status == SimulationStatus.READY:
            result["run_instructions"] = manager.get_run_instructions(simulation_id)

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"获取Simulaciónestado失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/list", methods=["GET"])
def list_simulations():
    """
    列出所有Simulación

    Query参数：
        project_id: 按proyectoID过滤（可选）
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
        logger.error(f"列出Simulación失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


def _get_report_id_for_simulation(simulation_id: str) -> str:
    """
    获取 simulation 对应的最新 report_id

    遍历 reports 目录，找出 simulation_id 匹配的 report，
    如果有多个则返回最新的（按 created_at 排序）

    Args:
        simulation_id: SimulaciónID

    Returns:
        report_id 或 None
    """
    import json
    from datetime import datetime

    # reports 目录路径：backend/uploads/reports
    # __file__ 是 app/api/simulation.py，需要向上两级到 backend/
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

        # 按创建时间倒序排序，返回最新的
        matching_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matching_reports[0].get("report_id")

    except Exception as e:
        logger.warning(f"查找 simulation {simulation_id} 的 report 失败: {e}")
        return None


@simulation_bp.route("/history", methods=["GET"])
def get_simulation_history():
    """
    获取historialSimulación列表（带proyecto详情）

    用于首页historialproyecto展示，返回包含proyecto名称、描述等丰富信息的Simulación列表

    Query参数：
        limit: 返回数量限制（默认20）

    返回：
        {
            "success": true,
            "data": [
                {
                    "simulation_id": "sim_xxxx",
                    "project_id": "proj_xxxx",
                    "project_name": "武大舆情分析",
                    "simulation_requirement": "如果武汉大学发布...",
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

        # 增强Simulación数据，只从 Simulation archivolectura
        enriched_simulations = []
        for sim in simulations:
            sim_dict = sim.to_dict()

            # 获取Simulaciónconfiguración信息（从 simulation_config.json lectura simulation_requirement）
            config = manager.get_simulation_config(sim.simulation_id)
            if config:
                sim_dict["simulation_requirement"] = config.get(
                    "simulation_requirement", ""
                )
                time_config = config.get("time_config", {})
                sim_dict["total_simulation_hours"] = time_config.get(
                    "total_simulation_hours", 0
                )
                # 推荐轮数（后备值）
                recommended_rounds = int(
                    time_config.get("total_simulation_hours", 0)
                    * 60
                    / max(time_config.get("minutes_per_round", 60), 1)
                )
            else:
                sim_dict["simulation_requirement"] = ""
                sim_dict["total_simulation_hours"] = 0
                recommended_rounds = 0

            # 获取运行estado（从 run_state.json lectura用户设置的实际轮数）
            run_state = SimulationRunner.get_run_state(sim.simulation_id)
            if run_state:
                sim_dict["current_round"] = run_state.current_round
                sim_dict["runner_status"] = run_state.runner_status.value
                # 使用用户设置的 total_rounds，若无则使用推荐轮数
                sim_dict["total_rounds"] = (
                    run_state.total_rounds
                    if run_state.total_rounds > 0
                    else recommended_rounds
                )
            else:
                sim_dict["current_round"] = 0
                sim_dict["runner_status"] = "idle"
                sim_dict["total_rounds"] = recommended_rounds

            # 获取关联proyecto的archivo列表（最多3个）
            project = ProjectManager.get_project(sim.project_id)
            if project and hasattr(project, "files") and project.files:
                sim_dict["files"] = [
                    {"filename": f.get("filename", "未知archivo")}
                    for f in project.files[:3]
                ]
            else:
                sim_dict["files"] = []

            # 获取关联的 report_id（查找该 simulation 最新的 report）
            sim_dict["report_id"] = _get_report_id_for_simulation(sim.simulation_id)

            # 添加版本号
            sim_dict["version"] = "v1.0.2"

            # 格式化日期
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
        logger.error(f"获取historialSimulación失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/profiles", methods=["GET"])
def get_simulation_profiles(simulation_id: str):
    """
    获取Simulación的Agent Profile

    Query参数：
        platform: 平台类型（reddit/twitter，默认reddit）
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
        logger.error(f"获取Profile失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/profiles/realtime", methods=["GET"])
def get_simulation_profiles_realtime(simulation_id: str):
    """
    tiempo real获取Simulación的Agent Profile（用于在generación过程中tiempo real查看progreso）

    与 /profiles 接口的区别：
    - 直接lecturaarchivo，不经过 SimulationManager
    - 适用于generación过程中的tiempo real查看
    - 返回额外的元数据（如archivo修改时间、是否正在generación等）

    Query参数：
        platform: 平台类型（reddit/twitter，默认reddit）

    返回：
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "platform": "reddit",
                "count": 15,
                "total_expected": 93,  // 预期总数（如果有）
                "is_generating": true,  // 是否正在generación
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

        # 获取Simulación目录
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify(
                {"success": False, "error": f"Simulación不存在: {simulation_id}"}
            ), 404

        # 确定archivo路径
        if platform == "reddit":
            profiles_file = os.path.join(sim_dir, "reddit_profiles.json")
        else:
            profiles_file = os.path.join(sim_dir, "twitter_profiles.csv")

        # 检查archivo是否存在
        file_exists = os.path.exists(profiles_file)
        profiles = []
        file_modified_at = None

        if file_exists:
            # 获取archivo修改时间
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
                logger.warning(f"lectura profiles archivo失败（可能正在写入中）: {e}")
                profiles = []

        # 检查是否正在generación（通过 state.json 判断）
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
        logger.error(f"tiempo real获取Profile失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config/realtime", methods=["GET"])
def get_simulation_config_realtime(simulation_id: str):
    """
    tiempo real获取Simulaciónconfiguración（用于在generación过程中tiempo real查看progreso）

    与 /config 接口的区别：
    - 直接lecturaarchivo，不经过 SimulationManager
    - 适用于generación过程中的tiempo real查看
    - 返回额外的元数据（如archivo修改时间、是否正在generación等）
    - 即使configuración还没generación完也能返回部分信息

    返回：
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "file_exists": true,
                "file_modified_at": "2025-12-04T18:20:00",
                "is_generating": true,  // 是否正在generación
                "generation_stage": "generating_config",  // 当前generaciónfase
                "config": {...}  // configuración内容（如果存在）
            }
        }
    """
    import json
    from datetime import datetime

    try:
        # 获取Simulación目录
        sim_dir = os.path.join(Config.OASIS_SIMULATION_DATA_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return jsonify(
                {"success": False, "error": f"Simulación不存在: {simulation_id}"}
            ), 404

        # configuraciónarchivo路径
        config_file = os.path.join(sim_dir, "simulation_config.json")

        # 检查archivo是否存在
        file_exists = os.path.exists(config_file)
        config = None
        file_modified_at = None

        if file_exists:
            # 获取archivo修改时间
            file_stat = os.stat(config_file)
            file_modified_at = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"lectura config archivo失败（可能正在写入中）: {e}")
                config = None

        # 检查是否正在generación（通过 state.json 判断）
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

                    # 判断当前fase
                    if is_generating:
                        if state_data.get("profiles_generated", False):
                            generation_stage = "generating_config"
                        else:
                            generation_stage = "generating_profiles"
                    elif status == "ready":
                        generation_stage = "completed"
            except Exception:
                pass

        # 构建返回数据
        response_data = {
            "simulation_id": simulation_id,
            "file_exists": file_exists,
            "file_modified_at": file_modified_at,
            "is_generating": is_generating,
            "generation_stage": generation_stage,
            "config_generated": config_generated,
            "config": config,
        }

        # 如果configuración存在，提取一些关键统计信息
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
        logger.error(f"tiempo real获取Config失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config", methods=["GET"])
def get_simulation_config(simulation_id: str):
    """
    获取Simulaciónconfiguración（LLM智能generación的完整configuración）

    返回包含：
        - time_config: 时间configuración（Simulación时长、轮次、高峰/低谷时段）
        - agent_configs: 每个Agent的活动configuración（活跃度、发言频率、立场等）
        - event_config: 事件configuración（初始帖子、热点话题）
        - platform_configs: 平台configuración
        - generation_reasoning: LLM的configuración推理说明
    """
    try:
        manager = SimulationManager()
        config = manager.get_simulation_config(simulation_id)

        if not config:
            return jsonify(
                {
                    "success": False,
                    "error": f"Simulaciónconfiguración不存在，请先调用 /prepare 接口",
                }
            ), 404

        return jsonify({"success": True, "data": config})

    except Exception as e:
        logger.error(f"获取configuración失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/config/download", methods=["GET"])
def download_simulation_config(simulation_id: str):
    """下载Simulaciónconfiguraciónarchivo"""
    try:
        manager = SimulationManager()
        sim_dir = manager._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            return jsonify(
                {
                    "success": False,
                    "error": "configuraciónarchivo不存在，请先调用 /prepare 接口",
                }
            ), 404

        return send_file(
            config_path, as_attachment=True, download_name="simulation_config.json"
        )

    except Exception as e:
        logger.error(f"下载configuración失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/script/<script_name>/download", methods=["GET"])
def download_simulation_script(script_name: str):
    """
    下载Simulación运行脚本archivo（通用脚本，位于 backend/scripts/）

    script_name可选值：
        - run_twitter_simulation.py
        - run_reddit_simulation.py
        - run_parallel_simulation.py
        - action_logger.py
    """
    try:
        # 脚本位于 backend/scripts/ 目录
        scripts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../scripts")
        )

        # 验证脚本名称
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
                    "error": f"未知脚本: {script_name}，可选: {allowed_scripts}",
                }
            ), 400

        script_path = os.path.join(scripts_dir, script_name)

        if not os.path.exists(script_path):
            return jsonify(
                {"success": False, "error": f"脚本archivo不存在: {script_name}"}
            ), 404

        return send_file(script_path, as_attachment=True, download_name=script_name)

    except Exception as e:
        logger.error(f"下载脚本失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Profilegeneración接口（独立使用） ==============


@simulation_bp.route("/generate-profiles", methods=["POST"])
def generate_profiles():
    """
    直接从graph_idgeneraciónOASIS Agent Profile（不创建Simulación）

    请求（JSON）：
        {
            "graph_id": "mirofish_xxxx",     // 必填
            "entity_types": ["Student"],      // 可选
            "use_llm": true,                  // 可选
            "platform": "reddit"              // 可选
        }
    """
    try:
        data = request.get_json() or {}

        graph_id = data.get("graph_id")
        if not graph_id:
            return jsonify({"success": False, "error": "请提供 graph_id"}), 400

        entity_types = data.get("entity_types")
        use_llm = data.get("use_llm", True)
        platform = data.get("platform", "reddit")

        reader = ZepEntityReader()
        filtered = reader.filter_defined_entities(
            graph_id=graph_id, defined_entity_types=entity_types, enrich_with_edges=True
        )

        if filtered.filtered_count == 0:
            return jsonify(
                {"success": False, "error": "没有找到符合条件的entidad"}
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
        logger.error(f"generaciónProfile失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Simulación运行控制接口 ==============


@simulation_bp.route("/start", methods=["POST"])
def start_simulation():
    """
    开始运行Simulación

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",          // 必填，SimulaciónID
            "platform": "parallel",                // 可选: twitter / reddit / parallel (默认)
            "max_rounds": 100,                     // 可选: 最大Simulación轮数，用于截断过长的Simulación
            "enable_graph_memory_update": false,   // 可选: 是否将Agent活动动态更新到Zepgraph_id记忆
            "force": false                         // 可选: 强制重新开始（会停止运行中的Simulación并清理日志）
        }

    关于 force 参数：
        - 启用后，如果Simulación正在运行或已completar，会先停止并清理运行日志
        - 清理的内容包括：run_state.json, actions.jsonl, simulation.log 等
        - 不会清理configuraciónarchivo（simulation_config.json）和 profile archivo
        - 适用于需要重新运行Simulación的场景

    关于 enable_graph_memory_update：
        - 启用后，Simulación中所有Agent的活动（发帖、评论、点赞等）都会tiempo real更新到Zepgraph_id
        - 这可以让graph_id"记住"Simulación过程，用于后续分析或AI对话
        - 需要Simulación关联的proyecto有有效的 graph_id
        - 采用批量更新机制，减少API调用次数

    返回：
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "runner_status": "running",
                "process_pid": 12345,
                "twitter_running": true,
                "reddit_running": true,
                "started_at": "2025-12-01T10:00:00",
                "graph_memory_update_enabled": true,  // 是否启用了graph_id记忆更新
                "force_restarted": true               // 是否是强制重新开始
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
        max_rounds = data.get("max_rounds")  # 可选：最大Simulación轮数
        enable_graph_memory_update = data.get(
            "enable_graph_memory_update", False
        )  # 可选：是否启用graph_id记忆更新
        force = data.get("force", False)  # 可选：强制重新开始

        # 验证 max_rounds 参数
        if max_rounds is not None:
            try:
                max_rounds = int(max_rounds)
                if max_rounds <= 0:
                    return jsonify(
                        {"success": False, "error": "max_rounds 必须是正整数"}
                    ), 400
            except (ValueError, TypeError):
                return jsonify(
                    {"success": False, "error": "max_rounds 必须是有效的整数"}
                ), 400

        if platform not in ["twitter", "reddit", "parallel"]:
            return jsonify(
                {
                    "success": False,
                    "error": f"无效的平台类型: {platform}，可选: twitter/reddit/parallel",
                }
            ), 400

        # 检查Simulación是否已preparar好
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)

        if not state:
            return jsonify(
                {"success": False, "error": f"Simulación不存在: {simulation_id}"}
            ), 404

        force_restarted = False

        # 智能处理estado：如果preparar工作已completar，允许重新iniciar
        if state.status != SimulationStatus.READY:
            # 检查preparar工作是否已completar
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)

            if is_prepared:
                # preparar工作已completar，检查是否有正在运行的进程
                if state.status == SimulationStatus.RUNNING:
                    # 检查Simulación进程是否真的在运行
                    run_state = SimulationRunner.get_run_state(simulation_id)
                    if run_state and run_state.runner_status.value == "running":
                        # 进程确实在运行
                        if force:
                            # 强制模式：停止运行中的Simulación
                            logger.info(
                                f"强制模式：停止运行中的Simulación {simulation_id}"
                            )
                            try:
                                SimulationRunner.stop_simulation(simulation_id)
                            except Exception as e:
                                logger.warning(f"停止Simulación时出现警告: {str(e)}")
                        else:
                            return jsonify(
                                {
                                    "success": False,
                                    "error": f"Simulación正在运行中，请先调用 /stop 接口停止，或使用 force=true 强制重新开始",
                                }
                            ), 400

                # 如果是强制模式，清理运行日志
                if force:
                    logger.info(f"强制模式：清理Simulación日志 {simulation_id}")
                    cleanup_result = SimulationRunner.cleanup_simulation_logs(
                        simulation_id
                    )
                    if not cleanup_result.get("success"):
                        logger.warning(
                            f"清理日志时出现警告: {cleanup_result.get('errors')}"
                        )
                    force_restarted = True

                # 进程不存在或已结束，重置estado为 ready
                logger.info(
                    f"Simulación {simulation_id} preparar工作已completar，重置estado为 ready（原estado: {state.status.value}）"
                )
                state.status = SimulationStatus.READY
                manager._save_simulation_state(state)
            else:
                # preparar工作未completar
                return jsonify(
                    {
                        "success": False,
                        "error": f"Simulación未preparar好，当前estado: {state.status.value}，请先调用 /prepare 接口",
                    }
                ), 400

        # 获取graph_idID（用于graph_id记忆更新）
        graph_id = None
        if enable_graph_memory_update:
            # 从Simulaciónestado或proyecto中获取 graph_id
            graph_id = state.graph_id
            if not graph_id:
                # 尝试从proyecto中获取
                project = ProjectManager.get_project(state.project_id)
                if project:
                    graph_id = project.graph_id

            if not graph_id:
                return jsonify(
                    {
                        "success": False,
                        "error": "启用graph_id记忆更新需要有效的 graph_id，请确保proyecto已构建graph_id",
                    }
                ), 400

            logger.info(
                f"启用graph_id记忆更新: simulation_id={simulation_id}, graph_id={graph_id}"
            )

        # iniciarSimulación
        run_state = SimulationRunner.start_simulation(
            simulation_id=simulation_id,
            platform=platform,
            max_rounds=max_rounds,
            enable_graph_memory_update=enable_graph_memory_update,
            graph_id=graph_id,
        )

        # 更新Simulaciónestado
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
        logger.error(f"iniciarSimulación失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/stop", methods=["POST"])
def stop_simulation():
    """
    停止Simulación

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx"  // 必填，SimulaciónID
        }

    返回：
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

        # 更新Simulaciónestado
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.PAUSED
            manager._save_simulation_state(state)

        return jsonify({"success": True, "data": run_state.to_dict()})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"停止Simulación失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== tiempo realestado监控接口 ==============


@simulation_bp.route("/<simulation_id>/run-status", methods=["GET"])
def get_run_status(simulation_id: str):
    """
    获取Simulación运行tiempo realestado（用于前端轮询）

    返回：
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
        logger.error(f"获取运行estado失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/run-status/detail", methods=["GET"])
def get_run_status_detail(simulation_id: str):
    """
    获取Simulación运行详细estado（包含所有动作）

    用于前端展示tiempo real动态

    Query参数：
        platform: 过滤平台（twitter/reddit，可选）

    返回：
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
                "twitter_actions": [...],  # Twitter 平台的所有动作
                "reddit_actions": [...]    # Reddit 平台的所有动作
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

        # 获取完整的动作列表
        all_actions = SimulationRunner.get_all_actions(
            simulation_id=simulation_id, platform=platform_filter
        )

        # 分平台获取动作
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

        # 获取当前轮次的动作（recent_actions 只展示最新一轮）
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

        # 获取基础estado信息
        result = run_state.to_dict()
        result["all_actions"] = [a.to_dict() for a in all_actions]
        result["twitter_actions"] = [a.to_dict() for a in twitter_actions]
        result["reddit_actions"] = [a.to_dict() for a in reddit_actions]
        result["rounds_count"] = len(run_state.rounds)
        # recent_actions 只展示当前最新一轮两个平台的内容
        result["recent_actions"] = [a.to_dict() for a in recent_actions]

        return jsonify({"success": True, "data": result})

    except Exception as e:
        logger.error(f"获取详细estado失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/actions", methods=["GET"])
def get_simulation_actions(simulation_id: str):
    """
    获取Simulación中的Agent动作historial

    Query参数：
        limit: 返回数量（默认100）
        offset: 偏移量（默认0）
        platform: 过滤平台（twitter/reddit）
        agent_id: 过滤Agent ID
        round_num: 过滤轮次

    返回：
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
        logger.error(f"获取动作historial失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/timeline", methods=["GET"])
def get_simulation_timeline(simulation_id: str):
    """
    获取Simulación时间线（按轮次汇总）

    用于前端展示progreso条和时间线视图

    Query参数：
        start_round: 起始轮次（默认0）
        end_round: 结束轮次（默认全部）

    返回每轮的汇总信息
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
        logger.error(f"获取时间线失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/agent-stats", methods=["GET"])
def get_agent_stats(simulation_id: str):
    """
    获取每个Agent的统计信息

    用于前端展示Agent活跃度排行、动作分布等
    """
    try:
        stats = SimulationRunner.get_agent_stats(simulation_id)

        return jsonify(
            {"success": True, "data": {"agents_count": len(stats), "stats": stats}}
        )

    except Exception as e:
        logger.error(f"获取Agent统计失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== 数据库consulta接口 ==============


@simulation_bp.route("/<simulation_id>/posts", methods=["GET"])
def get_simulation_posts(simulation_id: str):
    """
    获取Simulación中的帖子

    Query参数：
        platform: 平台类型（twitter/reddit）
        limit: 返回数量（默认50）
        offset: 偏移量

    返回帖子列表（从SQLite数据库lectura）
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
                        "message": "数据库不存在，Simulación可能尚未运行",
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
        logger.error(f"获取帖子失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/<simulation_id>/comments", methods=["GET"])
def get_simulation_comments(simulation_id: str):
    """
    获取Simulación中的评论（仅Reddit）

    Query参数：
        post_id: 过滤帖子ID（可选）
        limit: 返回数量
        offset: 偏移量
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
        logger.error(f"获取评论失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interview 采访接口 ==============


@simulation_bp.route("/interview", methods=["POST"])
def interview_agent():
    """
    采访单个Agent

    注意：此功能需要Simulación环境处于运行estado（completarSimulación循环后进入等待命令模式）

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",       // 必填，SimulaciónID
            "agent_id": 0,                     // 必填，Agent ID
            "prompt": "你对这件事有什么看法？",  // 必填，采访问题
            "platform": "twitter",             // 可选，指定平台（twitter/reddit）
                                               // 不指定时：双平台Simulación同时采访两个平台
            "timeout": 60                      // 可选，超时时间（秒），默认60
        }

    返回（不指定platform，双平台模式）：
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "你对这件事有什么看法？",
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

    返回（指定platform）：
        {
            "success": true,
            "data": {
                "agent_id": 0,
                "prompt": "你对这件事有什么看法？",
                "result": {
                    "agent_id": 0,
                    "response": "我认为...",
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
        platform = data.get("platform")  # 可选：twitter/reddit/None
        timeout = data.get("timeout", 60)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        if agent_id is None:
            return jsonify({"success": False, "error": "请提供 agent_id"}), 400

        if not prompt:
            return jsonify(
                {"success": False, "error": "请提供 prompt（采访问题）"}
            ), 400

        # 验证platform参数
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {"success": False, "error": "platform 参数只能是 'twitter' 或 'reddit'"}
            ), 400

        # 检查环境estado
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify(
                {
                    "success": False,
                    "error": "Simulación环境未运行或已关闭。请确保Simulación已completar并进入等待命令模式。",
                }
            ), 400

        # 优化prompt，添加前缀避免Agent调用工具
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
            {"success": False, "error": f"等待Interview响应超时: {str(e)}"}
        ), 504

    except Exception as e:
        logger.error(f"Interview失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/batch", methods=["POST"])
def interview_agents_batch():
    """
    批量采访多个Agent

    注意：此功能需要Simulación环境处于运行estado

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",       // 必填，SimulaciónID
            "interviews": [                    // 必填，采访列表
                {
                    "agent_id": 0,
                    "prompt": "你对A有什么看法？",
                    "platform": "twitter"      // 可选，指定该Agent的采访平台
                },
                {
                    "agent_id": 1,
                    "prompt": "你对B有什么看法？"  // 不指定platform则使用默认值
                }
            ],
            "platform": "reddit",              // 可选，默认平台（被每项的platform覆盖）
                                               // 不指定时：双平台Simulación每个Agent同时采访两个平台
            "timeout": 120                     // 可选，超时时间（秒），默认120
        }

    返回：
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
        platform = data.get("platform")  # 可选：twitter/reddit/None
        timeout = data.get("timeout", 120)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        if not interviews or not isinstance(interviews, list):
            return jsonify(
                {"success": False, "error": "请提供 interviews（采访列表）"}
            ), 400

        # 验证platform参数
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {"success": False, "error": "platform 参数只能是 'twitter' 或 'reddit'"}
            ), 400

        # 验证每个采访项
        for i, interview in enumerate(interviews):
            if "agent_id" not in interview:
                return jsonify(
                    {"success": False, "error": f"采访列表第{i + 1}项缺少 agent_id"}
                ), 400
            if "prompt" not in interview:
                return jsonify(
                    {"success": False, "error": f"采访列表第{i + 1}项缺少 prompt"}
                ), 400
            # 验证每项的platform（如果有）
            item_platform = interview.get("platform")
            if item_platform and item_platform not in ("twitter", "reddit"):
                return jsonify(
                    {
                        "success": False,
                        "error": f"采访列表第{i + 1}项的platform只能是 'twitter' 或 'reddit'",
                    }
                ), 400

        # 检查环境estado
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify(
                {
                    "success": False,
                    "error": "Simulación环境未运行或已关闭。请确保Simulación已completar并进入等待命令模式。",
                }
            ), 400

        # 优化每个采访项的prompt，添加前缀避免Agent调用工具
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
            {"success": False, "error": f"等待批量Interview响应超时: {str(e)}"}
        ), 504

    except Exception as e:
        logger.error(f"批量Interview失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/all", methods=["POST"])
def interview_all_agents():
    """
    全局采访 - 使用相同问题采访所有Agent

    注意：此功能需要Simulación环境处于运行estado

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",            // 必填，SimulaciónID
            "prompt": "你对这件事整体有什么看法？",  // 必填，采访问题（所有Agent使用相同问题）
            "platform": "reddit",                   // 可选，指定平台（twitter/reddit）
                                                    // 不指定时：双平台Simulación每个Agent同时采访两个平台
            "timeout": 180                          // 可选，超时时间（秒），默认180
        }

    返回：
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
        platform = data.get("platform")  # 可选：twitter/reddit/None
        timeout = data.get("timeout", 180)

        if not simulation_id:
            return jsonify(
                {"success": False, "error": "Por favor proporcione simulation_id"}
            ), 400

        if not prompt:
            return jsonify(
                {"success": False, "error": "请提供 prompt（采访问题）"}
            ), 400

        # 验证platform参数
        if platform and platform not in ("twitter", "reddit"):
            return jsonify(
                {"success": False, "error": "platform 参数只能是 'twitter' 或 'reddit'"}
            ), 400

        # 检查环境estado
        if not SimulationRunner.check_env_alive(simulation_id):
            return jsonify(
                {
                    "success": False,
                    "error": "Simulación环境未运行或已关闭。请确保Simulación已completar并进入等待命令模式。",
                }
            ), 400

        # 优化prompt，添加前缀避免Agent调用工具
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
            {"success": False, "error": f"等待全局Interview响应超时: {str(e)}"}
        ), 504

    except Exception as e:
        logger.error(f"全局Interview失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/interview/history", methods=["POST"])
def get_interview_history():
    """
    获取Interviewhistorial记录

    从Simulación数据库中lectura所有Interview记录

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",  // 必填，SimulaciónID
            "platform": "reddit",          // 可选，平台类型（reddit/twitter）
                                           // 不指定则返回两个平台的所有historial
            "agent_id": 0,                 // 可选，只获取该Agent的采访historial
            "limit": 100                   // 可选，返回数量，默认100
        }

    返回：
        {
            "success": true,
            "data": {
                "count": 10,
                "history": [
                    {
                        "agent_id": 0,
                        "response": "我认为...",
                        "prompt": "你对这件事有什么看法？",
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
        platform = data.get("platform")  # 不指定则返回两个平台的historial
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
        logger.error(f"获取Interviewhistorial失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/env-status", methods=["POST"])
def get_env_status():
    """
    获取Simulación环境estado

    检查Simulación环境是否存活（可以接收Interview命令）

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx"  // 必填，SimulaciónID
        }

    返回：
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "env_alive": true,
                "twitter_available": true,
                "reddit_available": true,
                "message": "环境正在运行，可以接收Interview命令"
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

        # 获取更详细的estado信息
        env_status = SimulationRunner.get_env_status_detail(simulation_id)

        if env_alive:
            message = "环境正在运行，可以接收Interview命令"
        else:
            message = "环境未运行或已关闭"

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
        logger.error(f"获取环境estado失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@simulation_bp.route("/close-env", methods=["POST"])
def close_simulation_env():
    """
    关闭Simulación环境

    向Simulación发送关闭环境命令，使其优雅退出等待命令模式。

    注意：这不同于 /stop 接口，/stop 会强制终止进程，
    而此接口会让Simulación优雅地关闭环境并退出。

    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",  // 必填，SimulaciónID
            "timeout": 30                  // 可选，超时时间（秒），默认30
        }

    返回：
        {
            "success": true,
            "data": {
                "message": "环境关闭命令已发送",
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

        # 更新Simulaciónestado
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        if state:
            state.status = SimulationStatus.COMPLETED
            manager._save_simulation_state(state)

        return jsonify({"success": result.get("success", False), "data": result})

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    except Exception as e:
        logger.error(f"关闭环境失败: {str(e)}")
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500
