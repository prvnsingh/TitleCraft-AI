"""
TitleCraft AI - Data Module

This module provides comprehensive functionality for handling, analyzing, and profiling
YouTube video data to support intelligent title generation.

Components:
- DataLoader: Load and preprocess video data
- DataValidator: Validate data quality and integrity  
- DataAnalyzer: Perform statistical analysis and extract insights
- ChannelProfiler: Create detailed channel profiles and patterns
- DataStore: Enhanced data management with caching and validation
- ChannelProfileManager: Manage channel profiles with automatic updates
- Models: Data structures and schemas
"""

from .models import VideoData, ChannelStats, ChannelProfile, TitlePatterns, DataQualityReport
from .loader import DataLoader
from .validator import DataValidator
from .analyzer import DataAnalyzer
from .profiler import ChannelProfiler
from .store import DataStore, ChannelProfileManager

__all__ = [
    # Data Models
    "VideoData",
    "ChannelStats", 
    "ChannelProfile",
    "TitlePatterns",
    "DataQualityReport",
    
    # Core Components
    "DataLoader",
    "DataValidator", 
    "DataAnalyzer",
    "ChannelProfiler",
    
    # Enhanced Components
    "DataStore",
    "ChannelProfileManager"
]

__version__ = "1.0.0"