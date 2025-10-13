"""
Production Monitoring and Logging System for TitleCraft AI
Implements structured logging, Prometheus metrics, health checks, and performance monitoring.
"""
import asyncio
import logging
import time
import psutil
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import threading
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.exposition import make_asgi_app
import structlog

from ..utils import BaseComponent

# Structured logging setup
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class MetricsCollector:
    """
    Prometheus metrics collector for TitleCraft AI.
    Tracks performance, usage, and system health metrics.
    """
    
    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        
        # API Metrics
        self.api_requests_total = Counter(
            'titlecraft_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'titlecraft_api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Title Generation Metrics
        self.title_generations_total = Counter(
            'titlecraft_title_generations_total',
            'Total number of title generations',
            ['channel_type', 'success'],
            registry=self.registry
        )
        
        self.title_generation_duration = Histogram(
            'titlecraft_title_generation_duration_seconds',
            'Title generation duration in seconds',
            ['llm_provider'],
            registry=self.registry
        )
        
        self.title_quality_scores = Histogram(
            'titlecraft_title_quality_scores',
            'Title quality score distribution',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # LLM Provider Metrics
        self.llm_requests_total = Counter(
            'titlecraft_llm_requests_total',
            'Total number of LLM requests',
            ['provider', 'status'],
            registry=self.registry
        )
        
        self.llm_request_duration = Histogram(
            'titlecraft_llm_request_duration_seconds',
            'LLM request duration in seconds',
            ['provider'],
            registry=self.registry
        )
        
        self.llm_tokens_used = Counter(
            'titlecraft_llm_tokens_used_total',
            'Total number of LLM tokens used',
            ['provider', 'type'],  # type: prompt, completion
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_operations_total = Counter(
            'titlecraft_cache_operations_total',
            'Total number of cache operations',
            ['operation', 'result'],  # operation: get, set, delete; result: hit, miss, success, error
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'titlecraft_cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )
        
        # System Health Metrics
        self.system_cpu_usage = Gauge(
            'titlecraft_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'titlecraft_system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_memory_available = Gauge(
            'titlecraft_system_memory_available_bytes',
            'System memory available in bytes',
            registry=self.registry
        )
        
        # Circuit Breaker Metrics
        self.circuit_breaker_state = Gauge(
            'titlecraft_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['name'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'titlecraft_circuit_breaker_failures_total',
            'Total number of circuit breaker failures',
            ['name'],
            registry=self.registry
        )
        
        # Data Processing Metrics
        self.channel_profiles_created = Counter(
            'titlecraft_channel_profiles_created_total',
            'Total number of channel profiles created',
            registry=self.registry
        )
        
        self.pattern_analyses_performed = Counter(
            'titlecraft_pattern_analyses_performed_total',
            'Total number of pattern analyses performed',
            registry=self.registry
        )

@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable
    timeout: float = 5.0
    critical: bool = True
    interval: float = 30.0
    last_check: Optional[datetime] = None
    last_status: bool = True
    consecutive_failures: int = 0

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

class MonitoringSystem(BaseComponent):
    """
    Comprehensive monitoring system for TitleCraft AI.
    Provides health checks, performance monitoring, and alerting.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.metrics_collector = MetricsCollector()
        self.logger = structlog.get_logger(__name__)
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status = True
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)  # Keep last 1000 data points
        self.request_times: deque = deque(maxlen=100)  # Last 100 request times
        self.error_count = 0
        self.total_requests = 0
        
        # System monitoring
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        self._setup_default_health_checks()

    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        async def check_cache_connection():
            """Check Redis cache connection."""
            try:
                from .cache import get_cache_manager
                cache = await get_cache_manager()
                await cache.redis_client.ping() if cache.redis_client else None
                return True
            except Exception:
                return False
                
        async def check_system_resources():
            """Check system resources are within acceptable limits."""
            try:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                return (cpu_percent < 90 and 
                       memory_percent < 90 and 
                       disk_percent < 90)
            except Exception:
                return False
                
        async def check_llm_providers():
            """Check LLM provider availability."""
            try:
                # This would be expanded to actually check each provider
                return True
            except Exception:
                return False
        
        self.health_checks = {
            'cache_connection': HealthCheck(
                name='cache_connection',
                check_function=check_cache_connection,
                critical=False
            ),
            'system_resources': HealthCheck(
                name='system_resources', 
                check_function=check_system_resources,
                critical=True
            ),
            'llm_providers': HealthCheck(
                name='llm_providers',
                check_function=check_llm_providers,
                critical=True
            )
        }

    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Monitoring system started")

    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Monitoring system stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Update system metrics
                await self._update_system_metrics()
                
                # Run health checks
                await self._run_health_checks()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep until next cycle
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(5)

    async def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_collector.system_memory_usage.set(memory.used)
            self.metrics_collector.system_memory_available.set(memory.available)
            
        except Exception as e:
            self.logger.warning("Failed to update system metrics", error=str(e))

    async def _run_health_checks(self):
        """Run all health checks."""
        overall_health = True
        
        for check_name, health_check in self.health_checks.items():
            try:
                # Skip if not enough time has passed
                if (health_check.last_check and 
                    (datetime.now() - health_check.last_check).total_seconds() < health_check.interval):
                    continue
                
                # Run the health check
                start_time = time.time()
                check_result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout
                )
                duration = time.time() - start_time
                
                health_check.last_check = datetime.now()
                
                if check_result:
                    health_check.last_status = True
                    health_check.consecutive_failures = 0
                    self.logger.debug(
                        "Health check passed",
                        check_name=check_name,
                        duration=duration
                    )
                else:
                    health_check.last_status = False
                    health_check.consecutive_failures += 1
                    self.logger.warning(
                        "Health check failed",
                        check_name=check_name,
                        consecutive_failures=health_check.consecutive_failures
                    )
                    
                    if health_check.critical:
                        overall_health = False
                        
            except asyncio.TimeoutError:
                health_check.last_status = False
                health_check.consecutive_failures += 1
                self.logger.error(
                    "Health check timeout",
                    check_name=check_name,
                    timeout=health_check.timeout
                )
                if health_check.critical:
                    overall_health = False
                    
            except Exception as e:
                health_check.last_status = False
                health_check.consecutive_failures += 1
                self.logger.error(
                    "Health check error",
                    check_name=check_name,
                    error=str(e)
                )
                if health_check.critical:
                    overall_health = False
        
        self.health_status = overall_health

    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            now = datetime.now()
            
            # Calculate current metrics
            if len(self.request_times) > 0:
                response_time_avg = sum(self.request_times) / len(self.request_times)
                sorted_times = sorted(self.request_times)
                response_time_p95 = sorted_times[int(0.95 * len(sorted_times))]
            else:
                response_time_avg = 0.0
                response_time_p95 = 0.0
            
            # Calculate requests per second (last minute)
            recent_requests = sum(1 for t in self.request_times if time.time() - t < 60)
            requests_per_second = recent_requests / 60.0
            
            # Calculate error rate
            error_rate = (self.error_count / max(self.total_requests, 1)) * 100
            
            # Get system metrics
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            # Create performance snapshot
            metrics = PerformanceMetrics(
                timestamp=now,
                response_time_avg=response_time_avg,
                response_time_p95=response_time_p95,
                requests_per_second=requests_per_second,
                error_rate=error_rate,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage
            )
            
            self.performance_history.append(metrics)
            
        except Exception as e:
            self.logger.warning("Failed to update performance metrics", error=str(e))

    def track_request(self, duration: float, success: bool = True):
        """Track API request performance."""
        self.request_times.append(time.time())
        self.total_requests += 1
        
        if not success:
            self.error_count += 1

    @asynccontextmanager
    async def track_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """Context manager for tracking operation performance."""
        labels = labels or {}
        start_time = time.time()
        
        try:
            yield
            duration = time.time() - start_time
            self.logger.info(
                "Operation completed",
                operation=operation_name,
                duration=duration,
                **labels
            )
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "Operation failed",
                operation=operation_name,
                duration=duration,
                error=str(e),
                **labels
            )
            raise

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        health_details = {}
        
        for check_name, health_check in self.health_checks.items():
            health_details[check_name] = {
                'status': 'healthy' if health_check.last_status else 'unhealthy',
                'last_check': health_check.last_check.isoformat() if health_check.last_check else None,
                'consecutive_failures': health_check.consecutive_failures,
                'critical': health_check.critical
            }
        
        return {
            'overall_health': 'healthy' if self.health_status else 'unhealthy',
            'checks': health_details,
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_summary(self, minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.performance_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {
                'period_minutes': minutes,
                'data_points': 0,
                'message': 'No data available for the requested period'
            }
        
        return {
            'period_minutes': minutes,
            'data_points': len(recent_metrics),
            'avg_response_time': sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics),
            'max_response_time': max(m.response_time_p95 for m in recent_metrics),
            'avg_requests_per_second': sum(m.requests_per_second for m in recent_metrics) / len(recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            'avg_cpu_usage': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_memory_usage': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'timestamp': datetime.now().isoformat()
        }

    def get_metrics_app(self):
        """Get Prometheus metrics ASGI app."""
        return make_asgi_app(registry=self.metrics_collector.registry)

# Global monitoring instance
_monitoring_system: Optional[MonitoringSystem] = None

def get_monitoring_system() -> MonitoringSystem:
    """Get or create global monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system

def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get structured logger instance."""
    return structlog.get_logger(name or __name__)

# Performance tracking decorators
def track_performance(operation_name: str, labels: Dict[str, str] = None):
    """Decorator to track function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitoring = get_monitoring_system()
            async with monitoring.track_operation(operation_name, labels):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'll track basic timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger = get_logger()
                logger.info(
                    "Operation completed",
                    operation=operation_name,
                    duration=duration,
                    **(labels or {})
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger = get_logger()
                logger.error(
                    "Operation failed", 
                    operation=operation_name,
                    duration=duration,
                    error=str(e),
                    **(labels or {})
                )
                raise
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator