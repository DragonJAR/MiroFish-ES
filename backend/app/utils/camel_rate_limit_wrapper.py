"""
Wrapper para rate limiting y timeout de simulaciones CAMEL-OASIS

Proporciona:
1. Wall-clock timeout — mata el subproceso si corre por demasiado tiempo
2. Circuit breaker — rastrea errores de ModelProcessingError
3. Rate limit detection — detecta patrones de rate limit

Este módulo解决 el problema de simulaciones CAMEL que nunca terminan porque
se quedan esperando en modo IPC comando-espera.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger("mirofish.camel_rate_limit")


class CircuitState(Enum):
    """Estados del circuit breaker"""

    CLOSED = "closed"  # Operación normal
    OPEN = "open"  # Rate limit detectado, bloqueando nuevas llamadas
    HALF_OPEN = "half_open"  # Probando si el rate limit se levantó


@dataclass
class CircuitBreakerConfig:
    """Configuración del circuit breaker"""

    failure_threshold: int = 5  # Abrir después de N fallas consecutivas
    recovery_timeout: float = 30.0  # Segundos antes de probar HALF_OPEN
    success_threshold: int = 2  # Cerrar después de N exitos en HALF_OPEN


@dataclass
class SimulationTimeoutConfig:
    """Configuración del timeout de simulación"""

    max_wall_clock_seconds: float = 3600.0  # Default 1 hora timeout
    check_interval: float = 10.0  # Cada cuánto verificar


class CamelRateLimitWrapper:
    """
    Wraps CAMEL subprocess simulation con:
    1. Wall-clock timeout — mata subprocess si corre demasiado
    2. Circuit breaker — rastrea ModelProcessingError
    3. Rate limit detection — detecta patrones de rate limit

    Uso:
        wrapper = CamelRateLimitWrapper(
            simulation_id=sim_id,
            process=subprocess_Popen,
            max_wall_clock_seconds=3600,
            on_timeout_callback=lambda: print("timeout!"),
            on_rate_limit_callback=lambda attempts: print(f"rate limit! attempts={attempts}")
        )
        wrapper.start()  # inicia monitoreo en thread background
        # ... simulación corre ...
        wrapper.stop()   # llamar cuando termina o cuando timeout dispara
    """

    def __init__(
        self,
        simulation_id: str,
        process,  # subprocess.Popen
        max_wall_clock_seconds: float = 3600.0,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        on_timeout_callback: Optional[Callable[[], None]] = None,
        on_rate_limit_callback: Optional[Callable[[int], None]] = None,
        on_circuit_open_callback: Optional[Callable[[], None]] = None,
    ):
        self.simulation_id = simulation_id
        self.process = process
        self.max_wall_clock_seconds = max_wall_clock_seconds

        # Circuit breaker config
        self.cb_config = circuit_breaker_config or CircuitBreakerConfig()

        # Callbacks
        self.on_timeout_callback = on_timeout_callback
        self.on_rate_limit_callback = on_rate_limit_callback
        self.on_circuit_open_callback = on_circuit_open_callback

        # Estado del circuit breaker (thread-safe)
        self._cb_lock = threading.Lock()
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._consecutive_rate_limits = 0

        # Estado del timeout
        self._timed_out = False
        self._timeout_fired = False
        self._start_time: Optional[float] = None

        # Control del thread de monitoreo
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def circuit_state(self) -> CircuitState:
        """Obtener estado actual del circuit breaker"""
        with self._cb_lock:
            return self._circuit_state

    @property
    def timed_out(self) -> bool:
        """True si el wall-clock timeout disparó"""
        return self._timed_out

    @property
    def failure_count(self) -> int:
        """Cantidad de fallas consecutivas"""
        with self._cb_lock:
            return self._failure_count

    def start(self):
        """Iniciar thread de monitoreo"""
        self._start_time = time.time()
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name=f"CamelRateLimit-{self.simulation_id}",
        )
        self._monitor_thread.start()
        logger.info(
            f"CamelRateLimit iniciado: simulation_id={self.simulation_id}, "
            f"timeout={self.max_wall_clock_seconds}s"
        )

    def stop(self):
        """Detener thread de monitoreo (completion normal)"""
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info(f"CamelRateLimit detenido: simulation_id={self.simulation_id}")

    def record_failure(self, error_type: str = "rate_limit"):
        """
        Registrar una falla (rate limit, ModelProcessingError, etc.)

        Si es un rate limit error y el circuit está HALF_OPEN → volver a OPEN
        Si las fallas exceed threshold → Abrir circuit
        """
        with self._cb_lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if error_type == "rate_limit":
                self._consecutive_rate_limits += 1
                logger.warning(
                    f"Rate limit detectado: simulation_id={self.simulation_id}, "
                    f"consecutive={self._consecutive_rate_limits}"
                )
                if self.on_rate_limit_callback:
                    self.on_rate_limit_callback(self._consecutive_rate_limits)
            else:
                self._consecutive_rate_limits = 0

            # Verificar si debemos abrir el circuit
            if self._circuit_state == CircuitState.HALF_OPEN:
                if error_type == "rate_limit":
                    # Rate limit en HALF_OPEN → volver a OPEN
                    self._circuit_state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit vuelto a OPEN por rate limit en HALF_OPEN: "
                        f"simulation_id={self.simulation_id}"
                    )
                    if self.on_circuit_open_callback:
                        self.on_circuit_open_callback()
                # Cualquier otra falla en HALF_OPEN también abre
                elif self._circuit_state == CircuitState.HALF_OPEN:
                    self._circuit_state = CircuitState.OPEN

            elif self._circuit_state == CircuitState.CLOSED:
                if self._failure_count >= self.cb_config.failure_threshold:
                    self._circuit_state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit OPEN (threshold alcanzado): "
                        f"simulation_id={self.simulation_id}, failures={self._failure_count}"
                    )
                    if self.on_circuit_open_callback:
                        self.on_circuit_open_callback()

    def record_success(self):
        """
        Registrar una llamada LLM exitosa

        Si está en HALF_OPEN y los exitos exceed threshold → cerrar circuit
        """
        with self._cb_lock:
            self._success_count += 1
            self._consecutive_rate_limits = 0

            if self._circuit_state == CircuitState.HALF_OPEN:
                if self._success_count >= self.cb_config.success_threshold:
                    self._circuit_state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(
                        f"Circuit CLOSED (recuperación exitosa): "
                        f"simulation_id={self.simulation_id}"
                    )

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (rate limit mode)"""
        with self._cb_lock:
            if self._circuit_state == CircuitState.OPEN:
                # Verificar si debemos probar HALF_OPEN
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.cb_config.recovery_timeout:
                        self._circuit_state = CircuitState.HALF_OPEN
                        self._success_count = 0
                        logger.info(
                            f"Circuit HALF_OPEN (timeout de recuperación): "
                            f"simulation_id={self.simulation_id}"
                        )
                        return False  # En HALF_OPEN permite intentos
                return True
            return False

    def get_state(self) -> Dict[str, Any]:
        """Retornar estado actual para debugging"""
        with self._cb_lock:
            return {
                "simulation_id": self.simulation_id,
                "circuit_state": self._circuit_state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "consecutive_rate_limits": self._consecutive_rate_limits,
                "timed_out": self._timed_out,
                "timeout_fired": self._timeout_fired,
                "elapsed_seconds": (
                    time.time() - self._start_time if self._start_time else 0
                ),
            }

    def _monitor_loop(self):
        """Loop principal de monitoreo (corre en thread separado)"""
        check_interval = 10.0  # Cada 10 segundos verificamos

        while not self._stop_event.is_set():
            try:
                # Verificar wall-clock timeout
                if self._start_time:
                    elapsed = time.time() - self._start_time
                    if elapsed >= self.max_wall_clock_seconds:
                        if not self._timeout_fired:
                            self._timeout_fired = True
                            self._timed_out = True
                            logger.error(
                                f"TIMEOUT de wall-clock: simulation_id={self.simulation_id}, "
                                f"elapsed={elapsed:.1f}s, limit={self.max_wall_clock_seconds}s"
                            )
                            if self.on_timeout_callback:
                                self.on_timeout_callback()
                    else:
                        # Log cada 5 minutos si aún no termina
                        if int(elapsed) % 300 == 0 and int(elapsed) > 0:
                            remaining = self.max_wall_clock_seconds - elapsed
                            logger.info(
                                f"Simulación en progreso: simulation_id={self.simulation_id}, "
                                f"elapsed={elapsed:.0f}s, remaining={remaining:.0f}s"
                            )

                # Verificar estado del circuit breaker
                self.is_circuit_open()

            except Exception as e:
                logger.warning(
                    f"Error en monitor loop: simulation_id={self.simulation_id}, error={e}"
                )

            # Esperar siguiente verificación
            self._stop_event.wait(check_interval)


class SimulationProcessManager:
    """
    Gestiona el ciclo de vida del subprocess CAMEL con timeout y circuit breaker.
    Esta es la API de alto nivel usada por SimulationRunner.

    Uso:
        manager = SimulationProcessManager(
            simulation_id=sim_id,
            process=subprocess_Popen,
            timeout_seconds=3600,
        )
        manager.start()
        # ...
        if manager.timed_out:
            print("Simulation timed out!")
        manager.stop()
    """

    def __init__(
        self,
        simulation_id: str,
        process,
        timeout_seconds: float = 3600.0,
        on_timeout: Optional[Callable[[], None]] = None,
        on_rate_limit: Optional[Callable[[int], None]] = None,
        on_circuit_open: Optional[Callable[[], None]] = None,
    ):
        self.simulation_id = simulation_id
        self.process = process
        self.timeout_seconds = timeout_seconds
        self.on_timeout_callback = on_timeout

        self._wrapper = CamelRateLimitWrapper(
            simulation_id=simulation_id,
            process=process,
            max_wall_clock_seconds=timeout_seconds,
            on_timeout_callback=on_timeout,
            on_rate_limit_callback=on_rate_limit,
            on_circuit_open_callback=on_circuit_open,
        )
        self._timeout_timer: Optional[threading.Timer] = None

    def start(self):
        """Iniciar el timeout timer y circuit breaker"""
        self._wrapper.start()
        self._timeout_timer = threading.Timer(self.timeout_seconds, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()
        logger.info(
            f"SimulationProcessManager iniciado: simulation_id={self.simulation_id}, "
            f"timeout={self.timeout_seconds}s"
        )

    def stop(self):
        """Detener timer y wrapper (completion normal)"""
        if self._timeout_timer:
            self._timeout_timer.cancel()
            self._timeout_timer = None
        self._wrapper.stop()
        logger.info(
            f"SimulationProcessManager detenido: simulation_id={self.simulation_id}"
        )

    def _on_timeout(self):
        """Called cuando el wall-clock timeout expira"""
        self._wrapper._timeout_fired = True
        self._wrapper._timed_out = True
        logger.error(
            f"TIMEOUT: simulation_id={self.simulation_id}, "
            f"timeout_seconds={self.timeout_seconds}s"
        )
        if self.on_timeout_callback:
            self.on_timeout_callback()

    def record_llm_failure(self, error_type: str = "rate_limit"):
        """Llamado por código externo cuando se detecta una falla LLM"""
        self._wrapper.record_failure(error_type)

    def record_llm_success(self):
        """Llamado por código externo cuando una llamada LLM succeeds"""
        self._wrapper.record_success()

    @property
    def circuit_state(self) -> CircuitState:
        return self._wrapper.circuit_state

    @property
    def timed_out(self) -> bool:
        return self._wrapper.timed_out

    def is_circuit_open(self) -> bool:
        return self._wrapper.is_circuit_open()

    def get_state(self) -> Dict[str, Any]:
        return self._wrapper.get_state()
