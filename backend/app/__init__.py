"""
MiroFish Backend - Flask应用工厂
"""

import os
import warnings

# 抑制 multiprocessing resource_tracker 的警告（来自第三方库如 transformers）
# 需要在所有其他导入之前设置
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger


def create_app(config_class=Config):
    """Flask应用工厂函数"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # 设置JSON编码：确保中文直接显示（而不是 \uXXXX 格式）
    # Flask >= 2.3 使用 app.json.ensure_ascii，旧版本使用 JSON_AS_ASCII 配置
    if hasattr(app, "json") and hasattr(app.json, "ensure_ascii"):
        app.json.ensure_ascii = False

    # 设置日志
    logger = setup_logger("mirofish")

    # 只在 reloader 子进程中打印启动信息（避免 debug 模式下打印两次）
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
