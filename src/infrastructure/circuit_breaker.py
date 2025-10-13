"""
Circuit Breaker and Error Handling System for TitleCraft AI
Implements robust error handling with graceful fallbacks and retry mechanisms.
"""
import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import traceback
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: int = 60          # Seconds before trying half-open
    success_threshold: int = 3          # Successes in half-open to close
    timeout: float = 30.0               # Request timeout in seconds
    expected_exceptions: tuple = (Exception,)  # Exceptions that trigger circuit

@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opened_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None

class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    Prevents cascading failures by temporarily blocking requests to failing services.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.stats = CircuitBreakerStats()
        
    async def call(self, func: Callable, *args, fallback: Callable = None, **kwargs) -> Any:
        """
        Execute function through circuit breaker with fallback support.
        
        Args:
            func: Function to execute
            *args: Function arguments
            fallback: Fallback function if main function fails
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
            
        Raises:
            CircuitOpenError: When circuit is open and no fallback provided
        """
        self.stats.total_requests += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_try_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN, blocking request")
                if fallback:
                    return await self._execute_fallback(fallback, *args, **kwargs)
                raise CircuitOpenError(f"Circuit breaker {self.name} is open")
        
        # Execute the function
        try:
            result = await asyncio.wait_for(
                self._execute_function(func, *args, **kwargs),
                timeout=self.config.timeout
            )
            
            self._record_success()
            return result
            
        except asyncio.TimeoutError:
            self.stats.timeouts += 1
            logger.warning(f"Timeout in circuit breaker {self.name}")
            self._record_failure()
            if fallback:
                return await self._execute_fallback(fallback, *args, **kwargs)
            raise
            
        except self.config.expected_exceptions as e:
            logger.warning(f"Expected exception in circuit breaker {self.name}: {e}")
            self._record_failure()
            if fallback:
                return await self._execute_fallback(fallback, *args, **kwargs)
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error in circuit breaker {self.name}: {e}")
            self._record_failure()
            if fallback:
                return await self._execute_fallback(fallback, *args, **kwargs)
            raise

    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the main function."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)

    async def _execute_fallback(self, fallback: Callable, *args, **kwargs) -> Any:
        """Execute fallback function."""
        try:
            logger.info(f"Executing fallback for circuit breaker {self.name}")
            if asyncio.iscoroutinefunction(fallback):
                return await fallback(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, fallback, *args, **kwargs)
        except Exception as e:
            logger.error(f"Fallback failed for circuit breaker {self.name}: {e}")
            raise

    def _record_success(self):
        """Record successful execution."""
        self.stats.successful_requests += 1
        self.stats.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._reset_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def _record_failure(self):
        """Record failed execution."""
        self.stats.failed_requests += 1
        self.stats.last_failure_time = datetime.now()
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            self._open_circuit()

    def _open_circuit(self):
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.stats.circuit_opened_count += 1
        logger.warning(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")

    def _reset_circuit(self):
        """Reset circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")

    def _should_try_reset(self) -> bool:
        """Check if enough time has passed to try resetting."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = (self.stats.successful_requests / self.stats.total_requests) * 100
            
        return {
            'name': self.name,
            'state': self.state.value,
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'success_rate': success_rate,
            'timeouts': self.stats.timeouts,
            'circuit_opened_count': self.stats.circuit_opened_count,
            'failure_count': self.failure_count,
            'last_failure_time': self.stats.last_failure_time,
            'last_success_time': self.stats.last_success_time,
        }

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class ErrorHandler:
    """
    Centralized error handling with categorization and recovery strategies.
    """
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies = {
            'api_timeout': self._handle_api_timeout,
            'api_rate_limit': self._handle_rate_limit,
            'api_auth_error': self._handle_auth_error,
            'data_validation_error': self._handle_validation_error,
            'llm_generation_error': self._handle_llm_error,
            'cache_error': self._handle_cache_error,
            'database_error': self._handle_database_error,
        }

    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle error with appropriate recovery strategy.
        
        Args:
            error: The exception to handle
            context: Additional context about the error
            
        Returns:
            Error handling result with recovery information
        """
        context = context or {}
        error_type = self._categorize_error(error)
        error_key = f"{error_type}_{str(type(error).__name__)}"
        
        # Track error frequency
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error details
        logger.error(f"Error handled: {error_type} - {str(error)}")
        logger.debug(f"Error context: {context}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Apply recovery strategy
        recovery_result = await self._apply_recovery_strategy(error_type, error, context)
        
        return {
            'error_type': error_type,
            'error_message': str(error),
            'error_count': self.error_counts[error_key],
            'recovery_applied': recovery_result['strategy_applied'],
            'recovery_success': recovery_result['success'],
            'fallback_data': recovery_result.get('fallback_data'),
            'retry_recommended': recovery_result.get('retry_recommended', False),
            'timestamp': datetime.now().isoformat(),
            'context': context,
        }

    def _categorize_error(self, error: Exception) -> str:
        """Categorize error type for appropriate handling."""
        error_name = type(error).__name__.lower()
        error_message = str(error).lower()
        
        if 'timeout' in error_name or 'timeout' in error_message:
            return 'api_timeout'
        elif 'rate limit' in error_message or 'too many requests' in error_message:
            return 'api_rate_limit'
        elif 'auth' in error_name or 'unauthorized' in error_message:
            return 'api_auth_error'
        elif 'validation' in error_name or 'invalid' in error_message:
            return 'data_validation_error'
        elif 'openai' in error_message or 'anthropic' in error_message:
            return 'llm_generation_error'
        elif 'redis' in error_message or 'cache' in error_message:
            return 'cache_error'
        elif 'database' in error_message or 'sql' in error_message:
            return 'database_error'
        else:
            return 'unknown_error'

    async def _apply_recovery_strategy(self, error_type: str, error: Exception, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply appropriate recovery strategy for error type."""
        strategy = self.recovery_strategies.get(error_type, self._handle_unknown_error)
        
        try:
            result = await strategy(error, context)
            result['strategy_applied'] = error_type
            return result
        except Exception as strategy_error:
            logger.error(f"Recovery strategy failed for {error_type}: {strategy_error}")
            return {
                'strategy_applied': 'none',
                'success': False,
                'retry_recommended': False,
                'error': str(strategy_error)
            }

    async def _handle_api_timeout(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API timeout errors."""
        return {
            'success': True,
            'retry_recommended': True,
            'retry_delay': 5,
            'fallback_data': self._generate_timeout_fallback(context),
            'message': 'API timeout handled with fallback data'
        }

    async def _handle_rate_limit(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rate limiting errors."""
        return {
            'success': True,
            'retry_recommended': True,
            'retry_delay': 60,  # Wait longer for rate limits
            'message': 'Rate limit detected, recommend retry with delay'
        }

    async def _handle_auth_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle authentication errors."""
        return {
            'success': False,
            'retry_recommended': False,
            'message': 'Authentication error - check API keys',
            'action_required': 'verify_credentials'
        }

    async def _handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation errors."""
        return {
            'success': True,
            'retry_recommended': False,
            'fallback_data': self._generate_validation_fallback(context),
            'message': 'Validation error handled with sanitized data'
        }

    async def _handle_llm_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM generation errors."""
        return {
            'success': True,
            'retry_recommended': True,
            'retry_delay': 10,
            'fallback_data': self._generate_llm_fallback(context),
            'message': 'LLM error handled with template-based fallback'
        }

    async def _handle_cache_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cache errors."""
        return {
            'success': True,
            'retry_recommended': False,
            'message': 'Cache error - continuing without cache',
            'bypass_cache': True
        }

    async def _handle_database_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database errors."""
        return {
            'success': False,
            'retry_recommended': True,
            'retry_delay': 30,
            'message': 'Database error - retry recommended'
        }

    async def _handle_unknown_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown errors."""
        return {
            'success': False,
            'retry_recommended': False,
            'message': f'Unknown error type: {type(error).__name__}'
        }

    def _generate_timeout_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback data for timeout errors."""
        return {
            'titles': [
                {
                    'title': 'Fallback Title - Service Temporarily Unavailable',
                    'confidence': 0.1,
                    'reasoning': 'Generated due to service timeout',
                    'fallback': True
                }
            ]
        }

    def _generate_validation_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback data for validation errors."""
        return {
            'sanitized_input': True,
            'validation_errors': ['Input data sanitized for processing'],
            'processed': True
        }

    def _generate_llm_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback data for LLM errors."""
        idea = context.get('idea', 'Unknown Topic')
        return {
            'titles': [
                {
                    'title': f'Amazing {idea} - Complete Guide',
                    'confidence': 0.3,
                    'reasoning': 'Template-based fallback due to LLM service error',
                    'fallback': True
                },
                {
                    'title': f'How to Master {idea} in 2024',
                    'confidence': 0.3,
                    'reasoning': 'Template-based fallback due to LLM service error',
                    'fallback': True
                }
            ]
        }

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error handling statistics."""
        total_errors = sum(self.error_counts.values())
        error_types = list(set([k.split('_')[0] for k in self.error_counts.keys()]))
        
        return {
            'total_errors_handled': total_errors,
            'unique_error_types': len(error_types),
            'error_breakdown': dict(self.error_counts),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None,
            'recovery_strategies_available': len(self.recovery_strategies)
        }

# Decorator for circuit breaker functionality
def circuit_breaker(name: str, config: CircuitBreakerConfig = None, fallback: Callable = None):
    """
    Decorator to add circuit breaker functionality to functions.
    
    Usage:
        @circuit_breaker('openai_api', fallback=fallback_function)
        async def call_openai_api():
            # API call here
            pass
    """
    def decorator(func):
        breaker = CircuitBreaker(name, config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, fallback=fallback, **kwargs)
        
        wrapper._circuit_breaker = breaker  # Expose breaker for monitoring
        return wrapper
    return decorator

# Global instances
_error_handler: Optional[ErrorHandler] = None
_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_error_handler() -> ErrorHandler:
    """Get or create global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get or create circuit breaker instance."""
    global _circuit_breakers
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]

def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all circuit breaker instances for monitoring."""
    return _circuit_breakers.copy()