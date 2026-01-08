"""
Circuit Breaker implementation for Phase 7.
Provides resilience patterns for external service calls.
"""
import time
import logging
import asyncio
from enum import Enum
from typing import Dict, Callable, Any, Optional, TypeVar, Awaitable
from dataclasses import dataclass, field
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    open_since: Optional[float] = None
    state_changes: int = 0


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""
    
    def __init__(self, circuit_name: str, retry_after: float):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{circuit_name}' is open. Retry after {retry_after:.1f}s")


class CircuitBreaker:
    """
    Circuit Breaker for resilient service calls.
    
    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: Service is failing, all calls rejected immediately
    - HALF_OPEN: Testing recovery, limited calls allowed
    
    Usage:
        cb = CircuitBreaker("llm_service")
        
        async def call_llm():
            return await cb.call(llm_client.generate, prompt)
        
        # Or as decorator:
        @cb.protect
        async def call_llm(prompt):
            return await llm_client.generate(prompt)
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._half_open_calls = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        if self._state != CircuitState.OPEN:
            return False
        
        # Check if timeout has elapsed
        if self._stats.open_since:
            elapsed = time.time() - self._stats.open_since
            if elapsed >= self.config.timeout_seconds:
                # Transition to half-open
                self._transition_to_half_open()
                return False
        
        return True
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def _transition_to_open(self) -> None:
        """Transition circuit to open state."""
        self._state = CircuitState.OPEN
        self._stats.open_since = time.time()
        self._stats.state_changes += 1
        logger.warning(f"Circuit '{self.name}' opened after {self._stats.consecutive_failures} failures")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._stats.consecutive_failures = 0
        self._stats.state_changes += 1
        logger.info(f"Circuit '{self.name}' transitioning to half-open")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._stats.consecutive_failures = 0
        self._stats.open_since = None
        self._half_open_calls = 0
        self._stats.state_changes += 1
        logger.info(f"Circuit '{self.name}' closed (service recovered)")
    
    def _record_success(self) -> None:
        """Record a successful call."""
        self._stats.total_calls += 1
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        self._stats.consecutive_failures = 0
        
        if self._state == CircuitState.HALF_OPEN:
            # Check if we've had enough successes to close
            if self._stats.successful_calls >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _record_failure(self) -> None:
        """Record a failed call."""
        self._stats.total_calls += 1
        self._stats.failed_calls += 1
        self._stats.consecutive_failures += 1
        self._stats.last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._transition_to_open()
        elif self._state == CircuitState.CLOSED:
            # Check if we've hit the failure threshold
            if self._stats.consecutive_failures >= self.config.failure_threshold:
                self._transition_to_open()
    
    def _should_allow_call(self) -> bool:
        """Check if a call should be allowed."""
        if self._state == CircuitState.CLOSED:
            return True
        
        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._stats.open_since:
                elapsed = time.time() - self._stats.open_since
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to_half_open()
                    return True
            return False
        
        if self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            return self._half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def _get_retry_after(self) -> float:
        """Get time until retry is allowed."""
        if self._stats.open_since:
            elapsed = time.time() - self._stats.open_since
            return max(0, self.config.timeout_seconds - elapsed)
        return 0
    
    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function through the circuit breaker.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitOpenError: If circuit is open
        """
        async with self._lock:
            if not self._should_allow_call():
                raise CircuitOpenError(self.name, self._get_retry_after())
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        
        except self.config.excluded_exceptions:
            # Don't count excluded exceptions as failures
            raise
        
        except Exception as e:
            self._record_failure()
            raise
    
    def protect(
        self,
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        """
        Decorator to protect an async function with this circuit breaker.
        
        Usage:
            @circuit_breaker.protect
            async def my_function():
                ...
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await self.call(func, *args, **kwargs)
        
        return wrapper
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        logger.info(f"Circuit '{self.name}' reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed circuit status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "consecutive_failures": self._stats.consecutive_failures,
                "state_changes": self._stats.state_changes,
            },
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            },
            "retry_after": self._get_retry_after() if self._state == CircuitState.OPEN else None,
        }


# ============================================================================
# CIRCUIT BREAKER REGISTRY
# ============================================================================

class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Provides a central place to configure and monitor all circuits.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._breakers: Dict[str, CircuitBreaker] = {}
        return cls._instance
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._breakers.get(name)
    
    def list_circuits(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return CircuitBreakerRegistry().get_or_create(name, config)


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return CircuitBreakerRegistry()


# Pre-configured circuit breakers for common services
LLM_CIRCUIT = get_circuit_breaker(
    "llm_service",
    CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=60.0,
        success_threshold=1,
    )
)

EMBEDDING_CIRCUIT = get_circuit_breaker(
    "embedding_service",
    CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=30.0,
        success_threshold=2,
    )
)

VECTOR_STORE_CIRCUIT = get_circuit_breaker(
    "vector_store",
    CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=30.0,
        success_threshold=2,
    )
)
