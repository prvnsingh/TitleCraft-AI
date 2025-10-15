"""
Simple FastAPI Application for TitleCraft AI Take-Home Task
Minimal implementation focusing on core requirements only
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

from src.services.title_generator import TitleGenerator
from src.data_module.data_processor import GeneratedTitle, DataLoader

# Request/Response Models
class TitleRequest(BaseModel):
    """Request model for title generation"""
    channel_id: str = Field(..., description="YouTube channel ID")
    idea: str = Field(..., description="Single-sentence video idea or topic")

class TitleResponse(BaseModel):
    """Individual title in response"""
    title: str = Field(..., description="Generated title")
    reasoning: str = Field(..., description="Data-grounded reasoning for this title")

class GenerationResponse(BaseModel):
    """Response model for title generation"""
    titles: List[TitleResponse] = Field(..., description="3-5 generated titles with reasoning")
    channel_id: str = Field(..., description="Channel ID processed")
    idea: str = Field(..., description="Video idea processed")

# Initialize FastAPI app
app = FastAPI(
    title="TitleCraft AI - Take-Home Task",
    description="YouTube title generation based on channel performance patterns",
    version="1.0.0"
)


@app.post("/generate", response_model=GenerationResponse)
async def generate_titles(request: TitleRequest) -> GenerationResponse:
    """
    Generate optimized YouTube titles based on channel patterns and video idea.
    
    This endpoint:
    1. Analyzes the channel's historical performance data
    2. Identifies patterns from high-performing titles
    3. Generates 3-5 new titles that follow successful patterns
    4. Provides data-grounded reasoning for each suggestion
    """
    # Initialize title generator (with fallback support)
    title_generator = TitleGenerator.__new__(TitleGenerator)
    title_generator.data_loader = DataLoader()
    
    # Check if OpenAI is available
    try:
        from openai import OpenAI
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            title_generator.client = OpenAI(api_key=api_key)
            use_openai = True
        else:
            title_generator.client = None
            use_openai = False
    except Exception:
        title_generator.client = None
        use_openai = False
    
    try:
        # Generate titles
        if use_openai:
            generated_titles = title_generator.generate_titles(
                channel_id=request.channel_id,
                idea=request.idea,
                n_titles=4
            )
        else:
            generated_titles = title_generator.generate_titles_fallback(
                channel_id=request.channel_id,
                idea=request.idea,
                n_titles=4
            )
        
        if not generated_titles:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for channel: {request.channel_id}"
            )
        
        # Format response
        response_titles = [
            TitleResponse(
                title=gen_title.title,
                reasoning=gen_title.reasoning
            ) for gen_title in generated_titles
        ]
        
        return GenerationResponse(
            titles=response_titles,
            channel_id=request.channel_id,
            idea=request.idea
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "TitleCraft AI",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "TitleCraft AI - Take-Home Task",
        "version": "1.0.0",
        "docs": "/docs",
        "generate_endpoint": "/generate",
        "description": "YouTube title generation based on channel performance patterns"
    }

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "src.apis.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )