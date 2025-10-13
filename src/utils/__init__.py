# Standard library imports
from typing import Optional, Any

# Local imports
from ..config import get_config


class ConfigurableMixin:
    """
    Mixin class providing standardized configuration handling.
    
    All classes that need configuration can inherit from this to get
    consistent config parameter handling.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize with optional config parameter"""
        self.config = config or get_config()


class LoggingMixin:
    """
    Mixin class providing standardized logging setup.
    
    Provides a logger instance configured with the class name.
    """
    
    @property
    def logger(self):
        """Get logger instance for this class"""
        if not hasattr(self, '_logger'):
            import logging
            self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger


class BaseComponent(ConfigurableMixin, LoggingMixin):
    """
    Base class for all TitleCraft AI components.
    
    Provides common configuration and logging functionality.
    All major classes should inherit from this for consistency.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize base component with config and logging"""
        super().__init__(config)
        
    def __repr__(self):
        """String representation of component"""
        return f"{self.__class__.__name__}(config={type(self.config).__name__})"