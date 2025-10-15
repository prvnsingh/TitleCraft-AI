"""
Simple FastAPI Application for TitleCraft AI Take-Home Task
Minimal implementation focusing on core requirements only
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn

try:
    # Try to use the enhanced title generator with LangChain
    from src.services.enhanced_title_generator import EnhancedTitleGenerator as TitleGenerator
    ENHANCED_MODE = True
except ImportError:
    # Fallback to original title generator
    from src.services.title_generator import TitleGenerator
    ENHANCED_MODE = False

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
    llm_info: Optional[Dict[str, Any]] = Field(None, description="Information about LLM service used")

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
    5. Uses enhanced LLM service with LangChain and tracing capabilities
    """
    try:
        # Initialize title generator with enhanced LLM service
        title_generator = TitleGenerator()
        
        # Generate titles using the enhanced service
        generated_titles = title_generator.generate_titles(
            channel_id=request.channel_id,
            idea=request.idea,
            n_titles=4
        )
        
        # Get service information
        llm_info = title_generator.get_service_info() if ENHANCED_MODE else None
        
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
            idea=request.idea,
            llm_info=llm_info
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Title generation failed: {str(e)}")

@app.get("/llm/status")
async def llm_status():
    """Get LLM service status and configuration"""
    try:
        if ENHANCED_MODE:
            from src.services.llm_config import validate_config
            generator = TitleGenerator()
            service_info = generator.get_service_info()
            config_status = validate_config()
            
            return {
                "enhanced_mode": True,
                "service_info": service_info,
                "config_status": config_status
            }
        else:
            return {
                "enhanced_mode": False,
                "message": "Using fallback mode - LangChain not available",
                "provider": "openai",
                "model": "gpt-3.5-turbo"
            }
    except Exception as e:
        return {
            "error": str(e),
            "enhanced_mode": False
        }

class ProviderSwitchRequest(BaseModel):
    """Request model for switching LLM provider"""
    provider: str = Field(..., description="LLM provider (openai, anthropic, ollama)")
    model: Optional[str] = Field(None, description="Model name (optional)")
    api_key: Optional[str] = Field(None, description="API key (optional)")

@app.post("/llm/switch")
async def switch_llm_provider(request: ProviderSwitchRequest):
    """Switch LLM provider"""
    if not ENHANCED_MODE:
        raise HTTPException(
            status_code=501, 
            detail="Provider switching not available in fallback mode"
        )
    
    try:
        generator = TitleGenerator()
        success = generator.switch_provider(
            provider=request.provider,
            model=request.model,
            api_key=request.api_key
        )
        
        if success:
            service_info = generator.get_service_info()
            return {
                "success": True,
                "message": f"Successfully switched to {request.provider}",
                "service_info": service_info
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Failed to switch provider"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Provider switch failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "TitleCraft AI",
        "version": "1.0.0",
        "enhanced_mode": ENHANCED_MODE
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