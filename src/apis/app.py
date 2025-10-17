"""
FastAPI Application for TitleCraft AI
Production-ready YouTube title generation with DeepSeek as default model
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from src.services.title_generator import (
    TitleGenerator,
    TitleGenerationRequest,
    TitleGenerationResponse,
)

# Constants
SERVICE_NAME = "TitleCraft AI"
SERVICE_VERSION = "2.0.0"
DEFAULT_MODEL = "DeepSeek-R1-Distill-Qwen-32B"


# Request/Response Models
class TitleRequest(BaseModel):
    """Request model for title generation"""

    channel_id: str = Field(..., description="YouTube channel ID")
    idea: str = Field(..., description="Single-sentence video idea or topic")
    n_titles: int = Field(4, description="Number of titles to generate (1-10)")
    temperature: Optional[float] = Field(
        None, description="Model temperature (0.0-1.0)"
    )
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")


class TitleResponseItem(BaseModel):
    """Individual title in response"""
    
    model_config = {"protected_namespaces": ()}

    title: str = Field(..., description="Generated title")
    reasoning: str = Field(..., description="Data-grounded reasoning for this title")
    confidence_score: float = Field(..., description="Confidence score (0.0-1.0)")
    model_used: str = Field(..., description="Model used to generate this title")


class GenerationResponse(BaseModel):
    """Response model for title generation"""
    
    model_config = {"protected_namespaces": ()}

    titles: List[TitleResponseItem] = Field(
        ..., description="Generated titles with reasoning"
    )
    channel_id: str = Field(..., description="Channel ID processed")
    idea: str = Field(..., description="Video idea processed")
    request_id: str = Field(..., description="Unique request ID for tracking")
    model_used: str = Field(..., description="LLM model used")
    provider: str = Field(..., description="LLM provider used")
    response_time: float = Field(..., description="Response time in seconds")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")
    success: bool = Field(..., description="Whether generation was successful")
    error_message: Optional[str] = Field(
        None, description="Error message if generation failed"
    )


class ModelInfo(BaseModel):
    """Model information"""
    
    model_config = {"protected_namespaces": ()}

    name: str
    provider: str
    model_name: str
    description: str
    use_case: str
    temperature: float
    max_tokens: int


class AvailableModelsResponse(BaseModel):
    """Response for available models"""

    models: Dict[str, ModelInfo] = Field(..., description="Available models")
    default_model: str = Field(..., description="Default model name")





# Initialize FastAPI app
app = FastAPI(
    title=SERVICE_NAME,
    description="YouTube title generation powered by DeepSeek AI",
    version=SERVICE_VERSION,
)

# Initialize the title generator with DeepSeek as default
title_generator = TitleGenerator()


@app.on_event("startup")
async def startup_event():
    """Log startup"""
    print(f"🚀 {SERVICE_NAME} started successfully")


@app.post("/generate", response_model=GenerationResponse)
async def generate_titles_default(request: TitleRequest) -> GenerationResponse:
    """
    Generate optimized YouTube titles using the default DeepSeek model.
    
    Args:
        request: TitleRequest with channel_id, idea, and optional parameters

    Returns:
        GenerationResponse: Generated titles with metadata
    """
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

    except Exception as e:
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
