"""
TitleCraft AI API Module - Minimal Implementation

Simple FastAPI application with:
- Core title generation endpoint
- Request/response validation with Pydantic
- Interactive API documentation
"""

from .production_app import app

__all__ = [
    "app",
]