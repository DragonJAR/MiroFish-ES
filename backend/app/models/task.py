"""
Gestión de estado de tareas
Se utiliza para rastrear tareas de larga duración (como la construcción de grafos)
"""

import uuid
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    """Enumeración de estados de tarea"""
    PENDING = "pending"          # Esperando
    PROCESSING = "processing"    # Procesando
    COMPLETED = "completed"      # Completado
    FAILED = "failed"            # Fallido


@dataclass
class Task:
    """Clase de datos de tarea"""
    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    progress: int = 0              # Porcentaje de progreso total 0-100
    message: str = ""              # Mensaje de estado
    result: Optional[Dict] = None  # Resultado de la tarea
    error: Optional[str] = None    # Información de error
    metadata: Dict = field(default_factory=dict)  # Metadatos adicionales
    progress_detail: Dict = field(default_factory=dict)  # Información detallada del progreso
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "message": self.message,
            "progress_detail": self.progress_detail,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskManager:
    """
    Gestor de tareas
    Gestión de estado de tareas thread-safe
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Patrón Singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, Task] = {}
                    cls._instance._task_lock = threading.Lock()
        return cls._instance
    
    def create_task(self, task_type: str, metadata: Optional[Dict] = None) -> str:
        """
        Crea una nueva tarea
        
        Args:
            task_type: Tipo de tarea
            metadata: Metadatos adicionales
            
        Returns:
            ID de tarea
        """
        task_id = str(uuid.uuid4())
        now = datetime.now()
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )
        
        with self._task_lock:
            self._tasks[task_id] = task
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Obtiene una tarea"""
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        progress_detail: Optional[Dict] = None
    ):
        """
        Actualiza el estado de la tarea
        
        Args:
            task_id: ID de tarea
            status: Nuevo estado
            progress: Progreso
            message: Mensaje
            result: Resultado
            error: Información de error
            progress_detail: Información detallada del progreso
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                task.updated_at = datetime.now()
                if status is not None:
                    task.status = status
                if progress is not None:
                    task.progress = progress
                if message is not None:
                    task.message = message
                if result is not None:
                    task.result = result
                if error is not None:
                    task.error = error
                if progress_detail is not None:
                    task.progress_detail = progress_detail
    
    def complete_task(self, task_id: str, result: Dict):
        """Marca la tarea como completada"""
        self.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Tarea completada",
            result=result
        )
    
    def fail_task(self, task_id: str, error: str):
        """Marca la tarea como fallida"""
        self.update_task(
            task_id,
            status=TaskStatus.FAILED,
            message="Tarea fallida",
            error=error
        )
    
    def list_tasks(self, task_type: Optional[str] = None) -> list:
        """Lista las tareas"""
        with self._task_lock:
            tasks = list(self._tasks.values())
            if task_type:
                tasks = [t for t in tasks if t.task_type == task_type]
            return [t.to_dict() for t in sorted(tasks, key=lambda x: x.created_at, reverse=True)]
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Limpia tareas antiguas"""
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._task_lock:
            old_ids = [
                tid for tid, task in self._tasks.items()
                if task.created_at < cutoff and task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
            ]
            for tid in old_ids:
                del self._tasks[tid]
