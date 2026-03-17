"""
Módulo de Comunicación IPC de Simulación
Utilizado para la comunicación entre procesos del backend Flask y los scripts de simulación

Implementa un simple patrón de comando/respuesta a través del sistema de archivos:
1. Flask escribe comandos al directorio commands/
2. El script de simulación sondea el directorio de comandos, ejecuta y escribe respuestas en responses/
3. Flask sondea el directorio de respuestas para obtener resultados
"""

import os
import json
import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger('mirofish.simulation_ipc')


class CommandType(str, Enum):
    """tipo"""
    INTERVIEW = "interview"           # Agent
    BATCH_INTERVIEW = "batch_interview"  # lote
    CLOSE_ENV = "close_env"           # 


class CommandStatus(str, Enum):
    """estado"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IPCCommand:
    """IPC"""
    command_id: str
    command_type: CommandType
    args: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type.value,
            "args": self.args,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCCommand':
        return cls(
            command_id=data["command_id"],
            command_type=CommandType(data["command_type"]),
            args=data.get("args", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class IPCResponse:
    """IPCrespuesta"""
    command_id: str
    status: CommandStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCResponse':
        return cls(
            command_id=data["command_id"],
            status=CommandStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )


class SimulationIPCClient:
    """
    simulaciónIPC（Flaskhacer）
    
    ensimulaciónprocesoenviaryrespuesta
    """
    
    def __init__(self, simulation_dir: str):
        """
        IPC
        
        Args:
            simulation_dir: simulacióndatosdirectorio
        """
        self.simulation_dir = simulation_dir
        self.commands_dir = os.path.join(simulation_dir, "ipc_commands")
        self.responses_dir = os.path.join(simulation_dir, "ipc_responses")
        
        # directorioen
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def send_command(
        self,
        command_type: CommandType,
        args: Dict[str, Any],
        timeout: float = 60.0,
        poll_interval: float = 0.5
    ) -> IPCResponse:
        """
        enviaryrespuesta
        
        Args:
            command_type: tipo
            args: parámetro
            timeout: timeouttiempo（）
            poll_interval: （）
            
        Returns:
            IPCResponse
            
        Raises:
            TimeoutError: respuestatimeout
        """
        command_id = str(uuid.uuid4())
        command = IPCCommand(
            command_id=command_id,
            command_type=command_type,
            args=args
        )
        
        # escribirarchivo
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        with open(command_file, 'w', encoding='utf-8') as f:
            json.dump(command.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"enviarIPC: {command_type.value}, command_id={command_id}")
        
        # respuesta
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(response_file):
                try:
                    with open(response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)
                    response = IPCResponse.from_dict(response_data)
                    
                    # yrespuestaarchivo
                    try:
                        os.remove(command_file)
                        os.remove(response_file)
                    except OSError:
                        pass
                    
                    logger.info(f"hastaIPCrespuesta: command_id={command_id}, status={response.status.value}")
                    return response
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"parsearrespuestafallido: {e}")
            
            time.sleep(poll_interval)
        
        # timeout
        logger.error(f"IPCrespuestatimeout: command_id={command_id}")
        
        # archivo
        try:
            os.remove(command_file)
        except OSError:
            pass
        
        raise TimeoutError(f"respuestatimeout ({timeout})")
    
    def send_interview(
        self,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> IPCResponse:
        """
        enviarAgent
        
        Args:
            agent_id: Agent ID
            prompt: 
            platform: （）
                - "twitter": Twitter
                - "reddit": Reddit  
                - None: simulación，simulación
            timeout: timeouttiempo
            
        Returns:
            IPCResponse，resultincluirresultado
        """
        args = {
            "agent_id": agent_id,
            "prompt": prompt
        }
        if platform:
            args["platform"] = platform
            
        return self.send_command(
            command_type=CommandType.INTERVIEW,
            args=args,
            timeout=timeout
        )
    
    def send_batch_interview(
        self,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0
    ) -> IPCResponse:
        """
        enviarlote
        
        Args:
            interviews: lista，incluir {"agent_id": int, "prompt": str, "platform": str()}
            platform: por defecto（，porplatform）
                - "twitter": por defectoTwitter
                - "reddit": por defectoReddit
                - None: simulaciónAgent
            timeout: timeouttiempo
            
        Returns:
            IPCResponse，resultincluirquetieneresultado
        """
        args = {"interviews": interviews}
        if platform:
            args["platform"] = platform
            
        return self.send_command(
            command_type=CommandType.BATCH_INTERVIEW,
            args=args,
            timeout=timeout
        )
    
    def send_close_env(self, timeout: float = 30.0) -> IPCResponse:
        """
        enviar
        
        Args:
            timeout: timeouttiempo
            
        Returns:
            IPCResponse
        """
        return self.send_command(
            command_type=CommandType.CLOSE_ENV,
            args={},
            timeout=timeout
        )
    
    def check_env_alive(self) -> bool:
        """
        simulaciónes
        
         env_status.json archivodeterminar
        """
        status_file = os.path.join(self.simulation_dir, "env_status.json")
        if not os.path.exists(status_file):
            return False
        
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            return status.get("status") == "alive"
        except (json.JSONDecodeError, OSError):
            return False


class SimulationIPCServer:
    """
    simulaciónIPC（simulaciónhacer）
    
    directorio，ejecutaryretornarrespuesta
    """
    
    def __init__(self, simulation_dir: str):
        """
        IPC
        
        Args:
            simulation_dir: simulacióndatosdirectorio
        """
        self.simulation_dir = simulation_dir
        self.commands_dir = os.path.join(simulation_dir, "ipc_commands")
        self.responses_dir = os.path.join(simulation_dir, "ipc_responses")
        
        # directorioen
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # estado
        self._running = False
    
    def start(self):
        """paraejecutarestado"""
        self._running = True
        self._update_env_status("alive")
    
    def stop(self):
        """paradetenerestado"""
        self._running = False
        self._update_env_status("stopped")
    
    def _update_env_status(self, status: str):
        """actualizarestadoarchivo"""
        status_file = os.path.join(self.simulation_dir, "env_status.json")
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_commands(self) -> Optional[IPCCommand]:
        """
        directorio，retornarprocesar
        
        Returns:
            IPCCommand o None
        """
        if not os.path.exists(self.commands_dir):
            return None
        
        # tiempoordenarobtenerarchivo
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return IPCCommand.from_dict(data)
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"leerarchivofallido: {filepath}, {e}")
                continue
        
        return None
    
    def send_response(self, response: IPCResponse):
        """
        enviarrespuesta
        
        Args:
            response: IPCrespuesta
        """
        response_file = os.path.join(self.responses_dir, f"{response.command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response.to_dict(), f, ensure_ascii=False, indent=2)
        
        # eliminararchivo
        command_file = os.path.join(self.commands_dir, f"{response.command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    def send_success(self, command_id: str, result: Dict[str, Any]):
        """enviaréxitorespuesta"""
        self.send_response(IPCResponse(
            command_id=command_id,
            status=CommandStatus.COMPLETED,
            result=result
        ))
    
    def send_error(self, command_id: str, error: str):
        """enviarerrorrespuesta"""
        self.send_response(IPCResponse(
            command_id=command_id,
            status=CommandStatus.FAILED,
            error=error
        ))
