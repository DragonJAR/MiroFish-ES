"""
Punto de entrada del backend MiroFish
"""

import os
import sys

# Resolver problema de caracteres codificados incorrectamente en consola Windows: establecer UTF-8 antes de todas las importaciones
if sys.platform == 'win32':
    # Establecer variable de entorno para que Python use UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # Reconfigurar flujos de salida estandar a UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Agregar directorio raiz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config


def main():
    """Funcion principal"""
    # Validar configuracion
    errors = Config.validate()
    if errors:
        print("Error de configuracion:")
        for err in errors:
            print(f"  - {err}")
        print("\nPor favor revise la configuracion en el archivo .env")
        sys.exit(1)
    
    # Crear aplicacion
    app = create_app()
    
    # Obtener configuracion de ejecucion
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    # Iniciar servicio
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()
