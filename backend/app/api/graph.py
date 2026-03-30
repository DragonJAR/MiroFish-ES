"""
API de rutas relacionadas con grafos
Utiliza mecanismo de contexto de proyecto, estado persistente en el servidor
"""

import os
import traceback
import threading
from flask import request, jsonify

from . import graph_bp
from ..config import Config
from ..services.ontology_generator import OntologyGenerator
from ..services.graph_builder import GraphBuilderService
from ..services.text_processor import TextProcessor
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from ..models.task import TaskManager, TaskStatus
from ..models.project import ProjectManager, ProjectStatus

# Obtener logger
logger = get_logger("mirofish.api")


def allowed_file(filename: str) -> bool:
    """Verificar si la extensión del archivo está permitida"""
    if not filename or "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext in Config.ALLOWED_EXTENSIONS


# ============== Interfaces de gestión de proyecto ==============


@graph_bp.route("/project/<project_id>", methods=["GET"])
def get_project(project_id: str):
    """
    Obtener detalles del proyecto
    """
    project = ProjectManager.get_project(project_id)

    if not project:
        return jsonify(
            {"success": False, "error": f"El proyecto no existe: {project_id}"}
        ), 404

    return jsonify({"success": True, "data": project.to_dict()})


@graph_bp.route("/project/list", methods=["GET"])
def list_projects():
    """
    Listar todos los proyectos
    """
    limit = request.args.get("limit", 50, type=int)
    projects = ProjectManager.list_projects(limit=limit)

    return jsonify(
        {
            "success": True,
            "data": [p.to_dict() for p in projects],
            "count": len(projects),
        }
    )


@graph_bp.route("/project/<project_id>", methods=["DELETE"])
def delete_project(project_id: str):
    """
    Eliminar proyecto
    """
    success = ProjectManager.delete_project(project_id)

    if not success:
        return jsonify(
            {
                "success": False,
                "error": f"El proyecto no existe o no se pudo eliminar: {project_id}",
            }
        ), 404

    return jsonify(
        {"success": True, "message": f"El proyecto ha sido eliminado: {project_id}"}
    )


@graph_bp.route("/project/<project_id>/reset", methods=["POST"])
def reset_project(project_id: str):
    """
    Restablecer estado del proyecto (para reconstruir el grafo)
    """
    project = ProjectManager.get_project(project_id)

    if not project:
        return jsonify(
            {"success": False, "error": f"El proyecto no existe: {project_id}"}
        ), 404

    # Restablecer al estado de ontología generada
    if project.ontology:
        project.status = ProjectStatus.ONTOLOGY_GENERATED
    else:
        project.status = ProjectStatus.CREATED

    project.graph_id = None
    project.graph_build_task_id = None
    project.error = None
    ProjectManager.save_project(project)

    return jsonify(
        {
            "success": True,
            "message": f"El proyecto ha sido restablecido: {project_id}",
            "data": project.to_dict(),
        }
    )


# ============== Interfaz 1: Subir archivo y generar ontología ==============


@graph_bp.route("/ontology/generate", methods=["POST"])
def generate_ontology():
    """
    Interfaz 1: Subir archivo, analizar y generar definición de ontología

    Tipo de solicitud: multipart/form-data

    Parámetros:
        files: Archivos subidos (PDF/MD/TXT), puede ser varios
        simulation_requirement: Descripción del requerimiento de simulación (obligatorio)
        project_name: Nombre del proyecto (opcional)
        additional_context: Contexto adicional (opcional)

    Retorna:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "ontology": {
                    "entity_types": [...],
                    "edge_types": [...],
                    "analysis_summary": "..."
                },
                "files": [...],
                "total_text_length": 12345
            }
        }
    """
    try:
        logger.info("=== Inicio de generación de ontología ===")

        # Obtener parámetros
        simulation_requirement = request.form.get("simulation_requirement", "")
        project_name = request.form.get("project_name", "Unnamed Project")
        additional_context = request.form.get("additional_context", "")

        logger.debug(f"Nombre del proyecto: {project_name}")
        logger.debug(f"Requerimiento de simulación: {simulation_requirement[:100]}...")

        if not simulation_requirement:
            return jsonify(
                {
                    "success": False,
                    "error": "Por favor proporciona la descripción del requerimiento de simulación (simulation_requirement)",
                }
            ), 400

        # Obtener archivos subidos
        uploaded_files = request.files.getlist("files")
        if not uploaded_files or all(not f.filename for f in uploaded_files):
            return jsonify(
                {"success": False, "error": "Por favor sube al menos un documento"}
            ), 400

        # Crear proyecto
        project = ProjectManager.create_project(name=project_name)
        project.simulation_requirement = simulation_requirement
        logger.info(f"Crear proyecto: {project.project_id}")

        # Guardar archivos y extraer texto
        document_texts = []
        all_text = ""

        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                # Guardar archivo en el directorio del proyecto
                file_info = ProjectManager.save_file_to_project(
                    project.project_id, file, file.filename
                )
                project.files.append(
                    {
                        "filename": file_info["original_filename"],
                        "size": file_info["size"],
                    }
                )

                # Extraer texto
                text = FileParser.extract_text(file_info["path"])
                text = TextProcessor.preprocess_text(text)
                document_texts.append(text)
                all_text += f"\n\n=== {file_info['original_filename']} ===\n{text}"

        if not document_texts:
            ProjectManager.delete_project(project.project_id)
            return jsonify(
                {
                    "success": False,
                    "error": "No se procesó ningún documento exitosamente, verifica el formato del archivo",
                }
            ), 400

        # Guardar texto extraído
        project.total_text_length = len(all_text)
        ProjectManager.save_extracted_text(project.project_id, all_text)
        logger.info(f"Extracción de texto completada, total {len(all_text)} caracteres")

        # Generar ontología
        logger.info("Llamando a LLM para generar ontología...")
        generator = OntologyGenerator()
        ontology = generator.generate(
            document_texts=document_texts,
            simulation_requirement=simulation_requirement,
            additional_context=additional_context if additional_context else None,
        )

        # Guardar ontología en el proyecto
        entity_count = len(ontology.get("entity_types", []))
        edge_count = len(ontology.get("edge_types", []))
        logger.info(
            f"Ontología generada: {entity_count} tipos de entidad, {edge_count} tipos de relación"
        )

        project.ontology = {
            "entity_types": ontology.get("entity_types", []),
            "edge_types": ontology.get("edge_types", []),
        }
        project.analysis_summary = ontology.get("analysis_summary", "")
        project.status = ProjectStatus.ONTOLOGY_GENERATED
        ProjectManager.save_project(project)
        logger.info(
            f"=== Generación de ontología completada === ID de proyecto: {project.project_id}"
        )

        return jsonify(
            {
                "success": True,
                "data": {
                    "project_id": project.project_id,
                    "project_name": project.name,
                    "ontology": project.ontology,
                    "analysis_summary": project.analysis_summary,
                    "files": project.files,
                    "total_text_length": project.total_text_length,
                },
            }
        )

    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz 2: Construir grafo ==============


@graph_bp.route("/build", methods=["POST"])
def build_graph():
    """
    Interfaz 2: Construir grafo según project_id

    Solicitud (JSON):
        {
            "project_id": "proj_xxxx",  // Obligatorio, de la interfaz 1
            "graph_name": "Nombre del grafo",    // Opcional
            "chunk_size": 500,          // Opcional, por defecto 500
            "chunk_overlap": 50         // Opcional, por defecto 50
        }

    Retorna:
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "task_id": "task_xxxx",
                "message": "Tarea de construcción del grafo iniciada"
            }
        }
    """
    try:
        logger.info("=== Inicio de construcción del grafo ===")

        # Verificar configuración
        errors = []
        if not Config.ZEP_API_KEY:
            errors.append("ZEP_API_KEY no configurado")
        if errors:
            logger.error(f"Error de configuración: {errors}")
            return jsonify(
                {
                    "success": False,
                    "error": "Error de configuración: " + "; ".join(errors),
                }
            ), 500

        # Parsear solicitud
        data = request.get_json() or {}
        project_id = data.get("project_id")
        logger.debug(f"Parámetros de solicitud: project_id={project_id}")

        if not project_id:
            return jsonify(
                {"success": False, "error": "Por favor proporciona project_id"}
            ), 400

        # Obtener proyecto
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify(
                {"success": False, "error": f"El proyecto no existe: {project_id}"}
            ), 404

        # Verificar estado del proyecto
        force = data.get("force", False)  # Reconstrucción forzada

        if project.status == ProjectStatus.CREATED:
            return jsonify(
                {
                    "success": False,
                    "error": "El proyecto no ha generado ontología, primero llama a /ontology/generate",
                }
            ), 400

        if project.status == ProjectStatus.GRAPH_BUILDING and not force:
            return jsonify(
                {
                    "success": False,
                    "error": "El grafo se está construyendo, no envíes de nuevo. Para reconstruir forzosamente, agrega force: true",
                    "task_id": project.graph_build_task_id,
                }
            ), 400

        # Si es reconstrucción forzada, restablecer estado
        if force and project.status in [
            ProjectStatus.GRAPH_BUILDING,
            ProjectStatus.FAILED,
            ProjectStatus.GRAPH_COMPLETED,
        ]:
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            project.graph_id = None
            project.graph_build_task_id = None
            project.error = None

        # Obtener configuración
        graph_name = data.get("graph_name", project.name or "MiroFish Graph")
        chunk_size = data.get(
            "chunk_size", project.chunk_size or Config.DEFAULT_CHUNK_SIZE
        )
        chunk_overlap = data.get(
            "chunk_overlap", project.chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP
        )

        # Actualizar configuración del proyecto
        project.chunk_size = chunk_size
        project.chunk_overlap = chunk_overlap

        # Obtener texto extraído
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            return jsonify(
                {
                    "success": False,
                    "error": "No se encontró contenido de texto extraído",
                }
            ), 400

        # Obtener ontología
        ontology = project.ontology
        if not ontology:
            return jsonify(
                {"success": False, "error": "No se encontró definición de ontología"}
            ), 400

        # Crear tarea asíncrona
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"Construir grafo: {graph_name}")
        logger.info(
            f"Crear tarea de construcción de grafo: task_id={task_id}, project_id={project_id}"
        )

        # Actualizar estado del proyecto
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        ProjectManager.save_project(project)

        # Iniciar tarea en segundo plano
        def build_task():
            build_logger = get_logger("mirofish.build")
            try:
                build_logger.info(f"[{task_id}] Inicio de construcción del grafo...")
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    message="Inicializando servicio de construcción de grafo...",
                )

                # Crear servicio de construcción de grafo
                builder = GraphBuilderService()

                # Fragmentar texto
                task_manager.update_task(
                    task_id, message="Fragmentando texto...", progress=5
                )
                chunks = TextProcessor.split_text(
                    text, chunk_size=chunk_size, overlap=chunk_overlap
                )
                total_chunks = len(chunks)

                # Crear grafo
                task_manager.update_task(
                    task_id, message="Creando grafo Zep...", progress=10
                )
                graph_id = builder.create_graph(name=graph_name)

                # Actualizar graph_id del proyecto
                project.graph_id = graph_id
                ProjectManager.save_project(project)

                # Configurar ontología
                task_manager.update_task(
                    task_id,
                    message="Configurando definición de ontología...",
                    progress=15,
                )
                builder.set_ontology(graph_id, ontology)

                # Agregar texto (progress_callback firma es (msg, progress_ratio))
                def add_progress_callback(msg, progress_ratio):
                    progress = 15 + int(progress_ratio * 40)  # 15% - 55%
                    task_manager.update_task(task_id, message=msg, progress=progress)

                task_manager.update_task(
                    task_id,
                    message=f"Inicio de agregar {total_chunks} fragmentos de texto...",
                    progress=15,
                )

                episode_uuids = builder.add_text_batches(
                    graph_id,
                    chunks,
                    batch_size=3,
                    progress_callback=add_progress_callback,
                )

                # Esperar a que Zep termine el procesamiento (consultar estado processed de cada episode)
                task_manager.update_task(
                    task_id,
                    message="Esperando procesamiento de datos por Zep...",
                    progress=55,
                )

                def wait_progress_callback(msg, progress_ratio):
                    progress = 55 + int(progress_ratio * 35)  # 55% - 90%
                    task_manager.update_task(task_id, message=msg, progress=progress)

                builder._wait_for_episodes(episode_uuids, wait_progress_callback)

                # Obtener datos del grafo
                task_manager.update_task(
                    task_id, message="Obteniendo datos del grafo...", progress=95
                )
                graph_data = builder.get_graph_data(graph_id)

                # Actualizar estado del proyecto
                project.status = ProjectStatus.GRAPH_COMPLETED
                ProjectManager.save_project(project)

                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)
                build_logger.info(
                    f"[{task_id}] Construcción del grafo completada: graph_id={graph_id}, nodos={node_count}, aristas={edge_count}"
                )

                # Completado
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    message="Construcción del grafo completada",
                    progress=100,
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "chunk_count": total_chunks,
                    },
                )

            except Exception as e:
                # Actualizar estado del proyecto a fallido
                build_logger.error(
                    f"[{task_id}] Construcción del grafo fallida: {str(e)}"
                )
                build_logger.debug(traceback.format_exc())

                project.status = ProjectStatus.FAILED
                project.error = str(e)
                ProjectManager.save_project(project)

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"Construcción fallida: {str(e)}",
                    error=traceback.format_exc(),
                )

        # Iniciar hilo en segundo plano
        thread = threading.Thread(target=build_task, daemon=True)
        thread.start()

        return jsonify(
            {
                "success": True,
                "data": {
                    "project_id": project_id,
                    "task_id": task_id,
                    "message": "Tarea de construcción del grafo iniciada, consulta el progreso en /task/{task_id}",
                },
            }
        )

    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


# ============== Interfaz de consulta de tareas ==============


@graph_bp.route("/task/<task_id>", methods=["GET"])
def get_task(task_id: str):
    """
    Consultar estado de tarea
    """
    task = TaskManager().get_task(task_id)

    if not task:
        return jsonify(
            {"success": False, "error": f"La tarea no existe: {task_id}"}
        ), 404

    return jsonify({"success": True, "data": task.to_dict()})


@graph_bp.route("/tasks", methods=["GET"])
def list_tasks():
    """
    Listar todas las tareas
    """
    tasks = TaskManager().list_tasks()

    return jsonify(
        {"success": True, "data": [t.to_dict() for t in tasks], "count": len(tasks)}
    )


# ============== Interfaz de datos del grafo ==============


@graph_bp.route("/data/<graph_id>", methods=["GET"])
def get_graph_data(graph_id: str):
    """
    Obtener datos del grafo (nodos y aristas)
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify(
                {"success": False, "error": "ZEP_API_KEY no configurado"}
            ), 500

        builder = GraphBuilderService()
        graph_data = builder.get_graph_data(graph_id)

        return jsonify({"success": True, "data": graph_data})

    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500


@graph_bp.route("/delete/<graph_id>", methods=["DELETE"])
def delete_graph(graph_id: str):
    """
    Eliminar grafo Zep
    """
    try:
        if not Config.ZEP_API_KEY:
            return jsonify(
                {"success": False, "error": "ZEP_API_KEY no configurado"}
            ), 500

        builder = GraphBuilderService()
        builder.delete_graph(graph_id)

        return jsonify({"success": True, "message": f"Grafo eliminado: {graph_id}"})

    except Exception as e:
        return jsonify(
            {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        ), 500
