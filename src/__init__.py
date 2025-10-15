"""
TitleCraft AI - Simple YouTube title generation system.

Minimal implementation for the take-home task focusing on core requirements:
- FastAPI endpoint for title generation
- Channel pattern analysis from CSV data
- LLM-based title generation with reasoning
"""

__version__ = "1.0.0"
__author__ = "TitleCraft AI Team"

# Import the main API app
try:
    from .api import app
    __all__ = ["app"]
except ImportError:
    __all__ = []