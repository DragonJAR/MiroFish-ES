"""
MiroFish Backend - Fábrica de aplicaciones Flask
"""

import os
import warnings

# Suprimir advertencias de multiprocessing resource_tracker (de librerías de terceros como transformers)
# Debe configurarse antes de todas las demás importaciones
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Función de fábrica de aplicaciones Flask"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configurar codificación JSON: asegurar que el chino se muestre directamente (en lugar de formato \uXXXX)
    # Flask >= 2.3 usa app.json.ensure_ascii, versiones anteriores usan configuración JSON_AS_ASCII
    if hasattr(app, "json") and hasattr(app.json, "ensure_ascii"):
        app.json.ensure_ascii = False

    # Configurar logging
    logger = setup_logger("mirofish")

    # Solo imprimir información de inicio en el subproceso del reloader (evitar duplicar en modo debug)
    is_reloader_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    debug_mode = app.config.get("DEBUG", False)
    should_log_startup = not debug_mode or is_reloader_process

    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend iniciando...")
        logger.info("=" * 50)

    # Habilitar CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Registrar funcion de limpieza de procesos de simulacion (al cerrar el servidor, terminar todos los procesos de simulacion)
    from .services.simulation_runner import SimulationRunner

    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Funcion de limpieza de procesos registrada")

    # Middleware de registro de solicitudes
    @app.before_request
    def log_request():
        logger = get_logger("mirofish.request")
        logger.debug(f"Solicitud: {request.method} {request.path}")
        if request.content_type and "json" in request.content_type:
            logger.debug(f"Cuerpo de solicitud: {request.get_json(silent=True)}")

    @app.after_request
    def log_response(response):
        logger = get_logger("mirofish.request")
        logger.debug(f"Respuesta: {response.status_code}")
        return response

    # Registrar blueprints
    from .api import graph_bp, simulation_bp, report_bp

    app.register_blueprint(graph_bp, url_prefix="/api/graph")
    app.register_blueprint(simulation_bp, url_prefix="/api/simulation")
    app.register_blueprint(report_bp, url_prefix="/api/report")

    # Verificacion de salud
    @app.route("/health")
    def health():
        return {"status": "ok", "service": "MiroFish Backend"}

    if should_log_startup:
        logger.info("MiroFish Backend iniciado correctamente")

    return app
