"""
Structured Logging System for TitleCraft AI
Creates three focused JSONL log files:
1. data_analytics.jsonl - Data analysis, metrics, pattern discovery, mathematical insights
2. llm_operations.jsonl - LLM interactions, prompt optimization, data injection
3. api_requests.jsonl - API request/response cycles, endpoint performance
"""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import functools
import time
import asyncio


class JSONLFormatter(logging.Formatter):
    """Formatter that outputs structured JSONL format"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add context
        for attr in ['request_id', 'channel_id', 'session_id', 'operation_id']:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
                
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class StructuredLogger:
    """Three-tier structured logging system for TitleCraft AI"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StructuredLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            StructuredLogger._initialized = True
    
    def _setup_logging(self):
        """Setup three specialized JSONL loggers"""
        # Create logs directory
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the three specialized loggers
        self._setup_data_analytics_logger()
        self._setup_llm_operations_logger()
        self._setup_api_requests_logger()
        
        # Log initialization
        self.log_data_analytics({
            "event": "logging_system_initialized",
            "log_directory": str(self.log_dir),
            "loggers": ["data_analytics", "llm_operations", "api_requests"]
        })
    
    def _setup_data_analytics_logger(self):
        """Setup data analytics logger for metrics, patterns, and mathematical insights"""
        self.data_logger = logging.getLogger("titlecraft.data_analytics")
        self.data_logger.setLevel(logging.INFO)
        self.data_logger.handlers.clear()
        
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "data_analytics.jsonl",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        handler.setFormatter(JSONLFormatter())
        self.data_logger.addHandler(handler)
    
    def _setup_llm_operations_logger(self):
        """Setup LLM operations logger for prompts, optimization, and data injection"""
        self.llm_logger = logging.getLogger("titlecraft.llm_operations")
        self.llm_logger.setLevel(logging.INFO)
        self.llm_logger.handlers.clear()
        
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "llm_operations.jsonl",
            maxBytes=30*1024*1024,  # 30MB
            backupCount=5
        )
        handler.setFormatter(JSONLFormatter())
        self.llm_logger.addHandler(handler)
    
    def _setup_api_requests_logger(self):
        """Setup API requests logger for endpoint performance and request/response cycles"""
        self.api_logger = logging.getLogger("titlecraft.api_requests")
        self.api_logger.setLevel(logging.INFO)
        self.api_logger.handlers.clear()
        
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "api_requests.jsonl",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=5
        )
        handler.setFormatter(JSONLFormatter())
        self.api_logger.addHandler(handler)
    
    # Data Analytics Logging Methods
    def log_data_analytics(self, data: Dict[str, Any], level: str = "INFO", **context):
        """Log data analytics events - metrics, patterns, mathematical insights"""
        extra_fields = {
            "category": "data_analytics",
            **data,
            **context
        }
        
        getattr(self.data_logger, level.lower())(
            data.get("message", "Data analytics event"),
            extra={'extra_fields': extra_fields, **context}
        )
    
    def log_pattern_analysis(self, patterns: Dict[str, Any], confidence: float, 
                           channel_data: Dict[str, Any], **context):
        """Log pattern discovery and analysis"""
        self.log_data_analytics({
            "event": "pattern_analysis",
            "patterns_discovered": patterns,
            "confidence_score": confidence,
            "channel_metrics": channel_data,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    def log_performance_metrics(self, metrics: Dict[str, Any], operation: str, **context):
        """Log performance and mathematical calculations"""
        self.log_data_analytics({
            "event": "performance_metrics",
            "operation": operation,
            "metrics": metrics,
            "calculation_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    def log_data_insights(self, insights: Dict[str, Any], data_source: str, **context):
        """Log data insights and contextual analysis"""
        self.log_data_analytics({
            "event": "data_insights",
            "data_source": data_source,
            "insights": insights,
            "context_generated": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    # LLM Operations Logging Methods
    def log_llm_operations(self, data: Dict[str, Any], level: str = "INFO", **context):
        """Log LLM operations events"""
        extra_fields = {
            "category": "llm_operations",
            **data,
            **context
        }
        
        getattr(self.llm_logger, level.lower())(
            data.get("message", "LLM operation event"),
            extra={'extra_fields': extra_fields, **context}
        )
    
    def log_prompt_construction(self, system_prompt: str, user_prompt: str, 
                              data_injected: Dict[str, Any], optimization_strategy: str, **context):
        """Log prompt construction and data injection"""
        self.log_llm_operations({
            "event": "prompt_construction",
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
            "system_prompt_preview": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt,
            "user_prompt_preview": user_prompt[:200] + "..." if len(user_prompt) > 200 else user_prompt,
            "data_injected": data_injected,
            "optimization_strategy": optimization_strategy,
            "prompt_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    def log_llm_request(self, model: str, parameters: Dict[str, Any], 
                       token_estimate: Optional[int] = None, **context):
        """Log LLM request details"""
        self.log_llm_operations({
            "event": "llm_request",
            "model": model,
            "parameters": parameters,
            "estimated_tokens": token_estimate,
            "request_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    def log_llm_response(self, response_text: str, tokens_used: Optional[int], 
                        cost: Optional[float], response_time: float, **context):
        """Log LLM response and performance"""
        self.log_llm_operations({
            "event": "llm_response",
            "response_length": len(response_text),
            "response_preview": response_text[:300] + "..." if len(response_text) > 300 else response_text,
            "tokens_used": tokens_used,
            "cost": cost,
            "response_time": response_time,
            "response_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    def log_prompt_optimization(self, original_strategy: str, optimized_strategy: str, 
                              optimization_reasons: Dict[str, Any], **context):
        """Log prompt optimization decisions"""
        self.log_llm_operations({
            "event": "prompt_optimization",
            "original_strategy": original_strategy,
            "optimized_strategy": optimized_strategy,
            "optimization_reasons": optimization_reasons,
            "optimization_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    # API Requests Logging Methods
    def log_api_requests(self, data: Dict[str, Any], level: str = "INFO", **context):
        """Log API request events"""
        extra_fields = {
            "category": "api_requests",
            **data,
            **context
        }
        
        getattr(self.api_logger, level.lower())(
            data.get("message", "API request event"),
            extra={'extra_fields': extra_fields, **context}
        )
    
    def log_api_request_start(self, endpoint: str, method: str, request_data: Dict[str, Any], 
                             client_info: Optional[Dict[str, Any]] = None, **context):
        """Log incoming API request"""
        self.log_api_requests({
            "event": "api_request_start",
            "endpoint": endpoint,
            "method": method,
            "request_data": request_data,
            "client_info": client_info or {},
            "request_start_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    def log_api_request_end(self, endpoint: str, method: str, status_code: int, 
                           response_data: Dict[str, Any], response_time: float, **context):
        """Log API request completion"""
        self.log_api_requests({
            "event": "api_request_end",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_size": len(str(response_data)),
            "response_summary": self._summarize_response(response_data),
            "response_time": response_time,
            "request_end_timestamp": datetime.now(timezone.utc).isoformat()
        }, **context)
    
    def log_api_error(self, endpoint: str, method: str, error: Exception, 
                     error_context: Dict[str, Any], **context):
        """Log API errors"""
        self.log_api_requests({
            "event": "api_error",
            "endpoint": endpoint,
            "method": method,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_context": error_context,
            "error_timestamp": datetime.now(timezone.utc).isoformat()
        }, level="ERROR", **context)
    
    def _summarize_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of response data"""
        summary = {}
        
        if "titles" in response_data:
            summary["titles_count"] = len(response_data["titles"])
            summary["has_titles"] = len(response_data["titles"]) > 0
        
        if "success" in response_data:
            summary["success"] = response_data["success"]
        
        if "tokens_used" in response_data:
            summary["tokens_used"] = response_data["tokens_used"]
        
        if "estimated_cost" in response_data:
            summary["estimated_cost"] = response_data["estimated_cost"]
        
        return summary
    
    # Compatibility methods for legacy logging calls
    def info(self, message: str, extra: Optional[Dict] = None):
        """Legacy compatibility for .info() calls"""
        if extra and 'extra_fields' in extra:
            component = extra['extra_fields'].get('component', 'unknown')
            if 'llm' in component or 'prompt' in component or 'context' in component:
                self.log_llm_operations({"event": "legacy_info", "message": message, **extra['extra_fields']})
            elif 'data' in component or 'pattern' in component or 'performance' in component:
                self.log_data_analytics({"event": "legacy_info", "message": message, **extra['extra_fields']})
            elif 'api' in component:
                self.log_api_requests({"event": "legacy_info", "message": message, **extra['extra_fields']})
            else:
                self.log_data_analytics({"event": "legacy_info", "message": message, **extra['extra_fields']})
        else:
            self.log_data_analytics({"event": "legacy_info", "message": message})
    
    def debug(self, message: str, extra: Optional[Dict] = None):
        """Legacy compatibility for .debug() calls"""
        if extra and 'extra_fields' in extra:
            component = extra['extra_fields'].get('component', 'unknown')
            if 'llm' in component or 'prompt' in component or 'context' in component:
                self.log_llm_operations({"event": "legacy_debug", "message": message, **extra['extra_fields']}, level="DEBUG")
            else:
                self.log_data_analytics({"event": "legacy_debug", "message": message, **extra['extra_fields']}, level="DEBUG")
        else:
            self.log_data_analytics({"event": "legacy_debug", "message": message}, level="DEBUG")
    
    def warning(self, message: str, extra: Optional[Dict] = None):
        """Legacy compatibility for .warning() calls"""
        if extra and 'extra_fields' in extra:
            self.log_data_analytics({"event": "legacy_warning", "message": message, **extra['extra_fields']}, level="WARNING")
        else:
            self.log_data_analytics({"event": "legacy_warning", "message": message}, level="WARNING")
    
    def error(self, message: str, extra: Optional[Dict] = None):
        """Legacy compatibility for .error() calls"""
        if extra and 'extra_fields' in extra:
            self.log_data_analytics({"event": "legacy_error", "message": message, **extra['extra_fields']}, level="ERROR")
        else:
            self.log_data_analytics({"event": "legacy_error", "message": message}, level="ERROR")


def log_data_operation(operation_name: str, component: str = "data"):
    """Decorator for data analytics operations"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger = structured_logger
                start_time = time.time()
                
                # Extract context
                context = _extract_context(args, kwargs)
                
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    logger.log_performance_metrics({
                        "operation": operation_name,
                        "component": component,
                        "execution_time": execution_time,
                        "success": True
                    }, operation_name, **context)
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.log_data_analytics({
                        "event": "operation_error",
                        "operation": operation_name,
                        "component": component,
                        "execution_time": execution_time,
                        "error": str(e),
                        "success": False
                    }, level="ERROR", **context)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger = structured_logger
                start_time = time.time()
                
                # Extract context
                context = _extract_context(args, kwargs)
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    logger.log_performance_metrics({
                        "operation": operation_name,
                        "component": component,
                        "execution_time": execution_time,
                        "success": True
                    }, operation_name, **context)
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.log_data_analytics({
                        "event": "operation_error",
                        "operation": operation_name,
                        "component": component,
                        "execution_time": execution_time,
                        "error": str(e),
                        "success": False
                    }, level="ERROR", **context)
                    raise
            return sync_wrapper
    return decorator


def log_llm_operation(operation_name: str, model: str = "unknown"):
    """Decorator for LLM operations"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger = structured_logger
                start_time = time.time()
                
                # Extract context
                context = _extract_context(args, kwargs)
                
                logger.log_llm_operations({
                    "event": f"{operation_name}_start",
                    "model": model,
                    "operation": operation_name
                }, **context)
                
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    logger.log_llm_operations({
                        "event": f"{operation_name}_success",
                        "model": model,
                        "operation": operation_name,
                        "execution_time": execution_time
                    }, **context)
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.log_llm_operations({
                        "event": f"{operation_name}_error",
                        "model": model,
                        "operation": operation_name,
                        "execution_time": execution_time,
                        "error": str(e)
                    }, level="ERROR", **context)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger = structured_logger
                start_time = time.time()
                
                # Extract context
                context = _extract_context(args, kwargs)
                
                logger.log_llm_operations({
                    "event": f"{operation_name}_start",
                    "model": model,
                    "operation": operation_name
                }, **context)
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    logger.log_llm_operations({
                        "event": f"{operation_name}_success",
                        "model": model,
                        "operation": operation_name,
                        "execution_time": execution_time
                    }, **context)
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.log_llm_operations({
                        "event": f"{operation_name}_error",
                        "model": model,
                        "operation": operation_name,
                        "execution_time": execution_time,
                        "error": str(e)
                    }, level="ERROR", **context)
                    raise
            return sync_wrapper
    return decorator


def log_api_operation(endpoint: str):
    """Decorator for API operations"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger = structured_logger
                start_time = time.time()
                
                # Extract request data from arguments
                request_data = {}
                if args and hasattr(args[0], 'dict'):  # Pydantic model
                    request_data = args[0].dict()
                elif 'request' in kwargs:
                    if hasattr(kwargs['request'], 'dict'):
                        request_data = kwargs['request'].dict()
                
                # Extract context
                context = _extract_context(args, kwargs)
                
                logger.log_api_request_start(
                    endpoint=endpoint,
                    method="POST",  # Assuming POST for most operations
                    request_data=request_data,
                    **context
                )
                
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Extract response data
                    response_data = result.dict() if hasattr(result, 'dict') else result
                    
                    logger.log_api_request_end(
                        endpoint=endpoint,
                        method="POST",
                        status_code=200,
                        response_data=response_data,
                        response_time=execution_time,
                        **context
                    )
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.log_api_error(
                        endpoint=endpoint,
                        method="POST",
                        error=e,
                        error_context={"execution_time": execution_time},
                        **context
                    )
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger = structured_logger
                start_time = time.time()
                
                # Extract request data from arguments
                request_data = {}
                if args and hasattr(args[0], 'dict'):  # Pydantic model
                    request_data = args[0].dict()
                elif 'request' in kwargs:
                    if hasattr(kwargs['request'], 'dict'):
                        request_data = kwargs['request'].dict()
                
                # Extract context
                context = _extract_context(args, kwargs)
                
                logger.log_api_request_start(
                    endpoint=endpoint,
                    method="POST",
                    request_data=request_data,
                    **context
                )
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Extract response data
                    response_data = result.dict() if hasattr(result, 'dict') else result
                    
                    logger.log_api_request_end(
                        endpoint=endpoint,
                        method="POST",
                        status_code=200,
                        response_data=response_data,
                        response_time=execution_time,
                        **context
                    )
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.log_api_error(
                        endpoint=endpoint,
                        method="POST",
                        error=e,
                        error_context={"execution_time": execution_time},
                        **context
                    )
                    raise
            return sync_wrapper
    return decorator


def _extract_context(args, kwargs):
    """Extract context information from function arguments"""
    context = {}
    
    # Try to extract request_id and channel_id
    if args:
        obj = args[0]
        for attr in ['current_request_id', 'request_id', 'channel_id']:
            if hasattr(obj, attr):
                context[attr] = getattr(obj, attr)
    
    # Check kwargs
    for key in ['request_id', 'channel_id', 'session_id']:
        if key in kwargs:
            context[key] = kwargs[key]
    
    return context


# Global structured logger instance
structured_logger = StructuredLogger()