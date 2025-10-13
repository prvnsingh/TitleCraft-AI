"""
TitleCraft AI API Module

Production FastAPI application with:
- RESTful title generation endpoints
- Request/response validation with Pydantic
- Rate limiting and authentication
- Interactive API documentation
- Comprehensive error handling
"""

from .production_app import create_app, app

__all__ = [
    "create_app",
    "app",
]