"""
Centralized Logging Configuration for TitleCraft AI

This module provides structured logging throughout the application with:
- File-based logging with rotation
- Structured JSON logs for analysis
- Context-aware log messages for code flow tracking
- Performance and reasoning insights
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import functools
import time


class TitleCraftFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs"""
    
    def format(self, record):
        # Create base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add execution context if available
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        if hasattr(record, 'channel_id'):
            log_entry['channel_id'] = record.channel_id
            
        if hasattr(record, 'performance_metrics'):
            log_entry['performance_metrics'] = record.performance_metrics
            
        return json.dumps(log_entry, ensure_ascii=False)


class TitleCraftLogger:
    """Centralized logger for TitleCraft AI application"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TitleCraftLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            TitleCraftLogger._initialized = True
    
    def _setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Create logs directory
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup main application logger
        self.logger = logging.getLogger("titlecraft")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Setup file handlers with rotation
        self._setup_file_handlers()
        
        # Setup console handler for development
        self._setup_console_handler()
        
        # Initial startup log
        self.logger.info("TitleCraft AI logging system initialized", extra={
            'extra_fields': {
                'component': 'logging_system',
                'action': 'initialization',
                'log_directory': str(self.log_dir)
            }
        })
    
    def _setup_file_handlers(self):
        """Setup rotating file handlers for different log types"""
        
        # Main application log with rotation
        app_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "titlecraft_app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(logging.INFO)
        app_handler.setFormatter(TitleCraftFormatter())
        
        # Debug log with detailed information
        debug_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "titlecraft_debug.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=3
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(TitleCraftFormatter())
        
        # Code flow and reasoning log
        flow_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "titlecraft_flow.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=5
        )
        flow_handler.setLevel(logging.INFO)
        flow_handler.setFormatter(TitleCraftFormatter())
        
        # Performance and analysis log
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "titlecraft_performance.log",
            maxBytes=30*1024*1024,  # 30MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(TitleCraftFormatter())
        
        # Add handlers to logger
        self.logger.addHandler(app_handler)
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(flow_handler)
        self.logger.addHandler(perf_handler)
    
    def _setup_console_handler(self):
        """Setup console handler for development"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        
        # Simple format for console
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """Get logger instance for specific module"""
        if name:
            return logging.getLogger(f"titlecraft.{name}")
        return self.logger


# Global logger instance
titlecraft_logger = TitleCraftLogger()


def log_execution_flow(operation_name: str, component: str = "unknown"):
    """
    Decorator to log function execution flow with performance metrics
    
    Args:
        operation_name: Name of the operation being performed
        component: Component/module name performing the operation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = titlecraft_logger.get_logger(component)
            
            # Extract context information
            request_id = getattr(args[0], 'current_request_id', None) if args else None
            channel_id = kwargs.get('channel_id') or (
                getattr(args[0], 'channel_id', None) if hasattr(args[0], 'channel_id') else None
            )
            
            start_time = time.time()
            
            # Log function entry
            logger.info(f"Starting {operation_name}", extra={
                'extra_fields': {
                    'component': component,
                    'operation': operation_name,
                    'action': 'start',
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                },
                'request_id': request_id,
                'channel_id': channel_id
            })
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log successful completion
                logger.info(f"Completed {operation_name}", extra={
                    'extra_fields': {
                        'component': component,
                        'operation': operation_name,
                        'action': 'complete',
                        'execution_time': execution_time,
                        'success': True
                    },
                    'request_id': request_id,
                    'channel_id': channel_id,
                    'performance_metrics': {
                        'execution_time': execution_time,
                        'operation': operation_name
                    }
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log error
                logger.error(f"Failed {operation_name}: {str(e)}", extra={
                    'extra_fields': {
                        'component': component,
                        'operation': operation_name,
                        'action': 'error',
                        'execution_time': execution_time,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    'request_id': request_id,
                    'channel_id': channel_id
                })
                
                raise
        
        return wrapper
    return decorator


def log_data_analysis(analysis_type: str, component: str = "data_analysis"):
    """
    Decorator specifically for data analysis and reasoning operations
    
    Args:
        analysis_type: Type of analysis being performed
        component: Component performing the analysis
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = titlecraft_logger.get_logger(component)
            
            start_time = time.time()
            
            # Log analysis start
            logger.info(f"Starting {analysis_type} analysis", extra={
                'extra_fields': {
                    'component': component,
                    'analysis_type': analysis_type,
                    'action': 'analysis_start'
                }
            })
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Extract analysis insights if result has specific attributes
                analysis_insights = {}
                if hasattr(result, '__dict__'):
                    # For dataclass objects, get relevant fields
                    for attr in ['patterns_found', 'confidence_score', 'metrics', 'strategy']:
                        if hasattr(result, attr):
                            analysis_insights[attr] = getattr(result, attr)
                elif isinstance(result, dict):
                    # For dict results, extract key metrics
                    for key in ['count', 'score', 'confidence', 'patterns']:
                        if key in result:
                            analysis_insights[key] = result[key]
                
                # Log analysis completion with insights
                logger.info(f"Completed {analysis_type} analysis", extra={
                    'extra_fields': {
                        'component': component,
                        'analysis_type': analysis_type,
                        'action': 'analysis_complete',
                        'execution_time': execution_time,
                        'analysis_insights': analysis_insights
                    }
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log analysis error
                logger.error(f"Failed {analysis_type} analysis: {str(e)}", extra={
                    'extra_fields': {
                        'component': component,
                        'analysis_type': analysis_type,
                        'action': 'analysis_error',
                        'execution_time': execution_time,
                        'error_type': type(e).__name__
                    }
                })
                
                raise
        
        return wrapper
    return decorator


def log_context_aware_decision(decision_type: str, reasoning: str, confidence: float, 
                              component: str = "context_aware"):
    """
    Log context-aware decisions with reasoning
    
    Args:
        decision_type: Type of decision being made
        reasoning: Reasoning behind the decision
        confidence: Confidence level in the decision
        component: Component making the decision
    """
    logger = titlecraft_logger.get_logger(component)
    
    logger.info(f"Context-aware decision: {decision_type}", extra={
        'extra_fields': {
            'component': component,
            'decision_type': decision_type,
            'reasoning': reasoning,
            'confidence': confidence,
            'action': 'context_decision'
        }
    })


def log_llm_interaction(model_name: str, prompt_type: str, tokens_used: Optional[int] = None, 
                       cost: Optional[float] = None, component: str = "llm_service"):
    """
    Log LLM interactions with performance metrics
    
    Args:
        model_name: Name of the model used
        prompt_type: Type of prompt used
        tokens_used: Number of tokens consumed
        cost: Estimated cost of the request
        component: Component handling the LLM interaction
    """
    logger = titlecraft_logger.get_logger(component)
    
    logger.info(f"LLM interaction with {model_name}", extra={
        'extra_fields': {
            'component': component,
            'model_name': model_name,
            'prompt_type': prompt_type,
            'tokens_used': tokens_used,
            'estimated_cost': cost,
            'action': 'llm_interaction'
        }
    })


# Export commonly used functions
__all__ = [
    'titlecraft_logger',
    'log_execution_flow',
    'log_data_analysis', 
    'log_context_aware_decision',
    'log_llm_interaction'
]