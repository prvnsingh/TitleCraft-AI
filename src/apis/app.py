"""
FastAPI Application for TitleCraft AI
Production-ready YouTube title generation with DeepSeek as default model
"""

from fastapi import FastAPI, HTTPException, Query
from typing import Any
import uvicorn

from src.services.title_generator import (
    TitleGenerator,
    TitleGenerationRequest,
    TitleGenerationResponse,
)
from src.apis.models import (
    TitleRequest,
    TitleResponseItem,
    GenerationResponse,
    ModelInfo,
    AvailableModelsResponse,
)
from src.services.logger_config import titlecraft_logger, log_execution_flow

# Constants
SERVICE_NAME = "TitleCraft AI"
SERVICE_VERSION = "2.0.0"
DEFAULT_MODEL = "DeepSeek-R1-Distill-Qwen-32B" # for reasoning-heavy tasks


# Initialize FastAPI app
app = FastAPI(
    title=SERVICE_NAME,
    description="YouTube title generation powered by DeepSeek AI",
    version=SERVICE_VERSION,
)

# Initialize logging for API
api_logger = titlecraft_logger.get_logger("api")

# Initialize the title generator with DeepSeek as default
title_generator = TitleGenerator()

api_logger.info("FastAPI application initialized", extra={
    'extra_fields': {
        'component': 'api',
        'action': 'app_initialization',
        'service_name': SERVICE_NAME,
        'service_version': SERVICE_VERSION,
        'default_model': DEFAULT_MODEL
    }
})



@app.post("/generate", response_model=GenerationResponse)
@log_execution_flow("api_generate_default", "api")
async def generate_titles_default(request: TitleRequest) -> GenerationResponse:
    """
    Generate optimized YouTube titles using the default DeepSeek model.
    
    Args:
        request: TitleRequest with channel_id, idea, and optional parameters

    Returns:
        GenerationResponse: Generated titles with metadata
    """
    api_logger.info("Received title generation request (default model)", extra={
        'extra_fields': {
            'component': 'api',
            'endpoint': '/generate',
            'action': 'request_received',
            'n_titles': request.n_titles,
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'idea_length': len(request.idea)
        },
        'channel_id': request.channel_id
    })
    
    try:
        # Create title generation request with default DeepSeek model
        gen_request = TitleGenerationRequest(
            channel_id=request.channel_id,
            video_idea=request.idea,
            n_titles=request.n_titles,
            model_name=DEFAULT_MODEL,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        api_logger.info("Calling title generator", extra={
            'extra_fields': {
                'component': 'api',
                'endpoint': '/generate',
                'action': 'calling_title_generator',
                'model_name': DEFAULT_MODEL
            },
            'channel_id': request.channel_id
        })

        # Generate titles
        response = title_generator.generate_titles(gen_request)

        api_logger.info("Title generation completed", extra={
            'extra_fields': {
                'component': 'api',
                'endpoint': '/generate',
                'action': 'generation_completed',
                'success': response.success,
                'titles_count': len(response.titles),
                'response_time': response.response_time,
                'tokens_used': response.tokens_used,
                'estimated_cost': response.estimated_cost
            },
            'channel_id': request.channel_id,
            'request_id': response.request_id
        })

        # Convert GeneratedTitle objects to TitleResponseItem
        title_items = [
            TitleResponseItem(
                title=title.title,
                reasoning=title.reasoning,
                confidence_score=getattr(title, "confidence_score", 0.8),
                model_used=getattr(title, "model_used", response.model_used),
            )
            for title in response.titles
        ]

        api_response = GenerationResponse(
            titles=title_items,
            channel_id=request.channel_id,
            idea=request.idea,
            request_id=response.request_id,
            model_used=response.model_used,
            provider=response.provider,
            response_time=response.response_time,
            tokens_used=response.tokens_used,
            estimated_cost=response.estimated_cost,
            success=response.success,
            error_message=response.error_message,
        )

        api_logger.info("API response prepared successfully", extra={
            'extra_fields': {
                'component': 'api',
                'endpoint': '/generate',
                'action': 'response_prepared'
            },
            'channel_id': request.channel_id,
            'request_id': response.request_id
        })

        return api_response

    except Exception as e:
        api_logger.error("Title generation failed in API", extra={
            'extra_fields': {
                'component': 'api',
                'endpoint': '/generate',
                'action': 'generation_error',
                'error_type': type(e).__name__,
                'error_message': str(e)
            },
            'channel_id': request.channel_id
        })
        
        raise HTTPException(
            status_code=500, detail=f"Title generation failed: {str(e)}"
        )


@app.post("/generate-with-model", response_model=GenerationResponse)
async def generate_titles_with_model(
    request: TitleRequest,
    model: str = Query(..., description="Model name to use for generation"),
) -> GenerationResponse:
    """
    Generate optimized YouTube titles with a specified model.
    Allows selection from available models when different from default DeepSeek.

    Args:
        request: TitleRequest with channel_id, idea, and optional parameters
        model: Model name from available models (use /models endpoint to see options)

    Returns:
        GenerationResponse: Generated titles with metadata
    """
    try:
        # Validate model is available
        available_models = title_generator.get_available_models()
        if model not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model}' not available. Use /models endpoint to see available models.",
            )

        # Create title generation request with specified model
        gen_request = TitleGenerationRequest(
            channel_id=request.channel_id,
            video_idea=request.idea,
            n_titles=request.n_titles,
            model_name=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        # Generate titles
        response = title_generator.generate_titles(gen_request)

        # Convert GeneratedTitle objects to TitleResponseItem
        title_items = [
            TitleResponseItem(
                title=title.title,
                reasoning=title.reasoning,
                confidence_score=getattr(title, "confidence_score", 0.8),
                model_used=getattr(title, "model_used", response.model_used),
            )
            for title in response.titles
        ]

        return GenerationResponse(
            titles=title_items,
            channel_id=request.channel_id,
            idea=request.idea,
            request_id=response.request_id,
            model_used=response.model_used,
            provider=response.provider,
            response_time=response.response_time,
            tokens_used=response.tokens_used,
            estimated_cost=response.estimated_cost,
            success=response.success,
            error_message=response.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Title generation failed: {str(e)}"
        )


@app.get("/models", response_model=AvailableModelsResponse)
async def get_available_models() -> AvailableModelsResponse:
    """
    Get list of available LLM models with their information.

    Returns:
        AvailableModelsResponse: Available models with metadata
    """
    try:
        available_models = title_generator.get_available_models()

        models = {}
        for name, info in available_models.items():
            models[name] = ModelInfo(
                name=name,
                provider=info["provider"],
                model_name=info["model_name"],
                description=info["description"],
                use_case=info["use_case"],
                temperature=info["temperature"],
                max_tokens=info["max_tokens"],
            )

        return AvailableModelsResponse(models=models, default_model=DEFAULT_MODEL)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "default_model": DEFAULT_MODEL,
        "available_models_count": len(title_generator.get_available_models()),
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "default_model": DEFAULT_MODEL,
        "docs": "/docs",
        "endpoints": {
            "generate_default": "/generate",
            "generate_with_model": "/generate-with-model",
            "available_models": "/models",
            "health": "/health",
        },
        "description": "YouTube title generation powered by DeepSeek AI",
    }


# Development server
if __name__ == "__main__":
    uvicorn.run("src.apis.app:app", host="0.0.0.0", port=8000, reload=True)
