"""
TitleCraft AI - Production-ready YouTube title generation system.

A comprehensive system for generating high-performance YouTube titles using
multi-model LLM orchestration, intelligent caching, and production monitoring.
"""

__version__ = "1.0.0"
__author__ = "TitleCraft AI Team"

# Core data models (always available)
from .data import ChannelProfile, VideoData

# Processing components (basic functionality)
from .processing import LLMOrchestrator

__all__ = [
    "ChannelProfile",
    "VideoData", 
    "LLMOrchestrator",
]

# Optional production components (require additional dependencies)
try:
    from .infrastructure import MultiLLMOrchestrator, CacheManager
    __all__.extend(["MultiLLMOrchestrator", "CacheManager"])
except ImportError:
    pass

try:
    from .api import create_app
    __all__.append("create_app")
except ImportError:
    pass