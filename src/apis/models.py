"""
Pydantic Models for TitleCraft AI API
Request and Response models for the FastAPI application
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


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