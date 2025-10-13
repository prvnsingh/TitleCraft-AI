"""
Production-Ready FastAPI Application for TitleCraft AI
Includes comprehensive validation, documentation, rate limiting, and monitoring.
"""
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import time
import uvicorn
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import os

# Import infrastructure components
from ..infrastructure.cache import get_cache_manager, shutdown_cache
from ..infrastructure.monitoring import get_monitoring_system, get_logger
from ..infrastructure.circuit_breaker import get_error_handler
from ..infrastructure.multi_llm import create_llm_orchestrator, LLMRequest

# Import existing services
from ..services.generation_engine import TitleGenerationEngine
from ..data.store import DataStore
from ..config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

# API Models
class TitleGenerationRequest(BaseModel):
    """Request model for title generation."""
    idea: str = Field(..., min_length=1, max_length=500, description="Video idea or topic")
    channel_id: str = Field(..., min_length=1, max_length=100, description="YouTube channel ID")
    n_titles: int = Field(default=4, ge=1, le=10, description="Number of titles to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature for creativity")
    max_tokens: int = Field(default=2000, ge=100, le=4000, description="Maximum tokens for generation")
    provider_preference: Optional[List[str]] = Field(default=None, description="Preferred LLM providers")
    include_reasoning: bool = Field(default=True, description="Include reasoning explanations")
    
    @validator('idea')
    def validate_idea(cls, v):
        if not v.strip():
            raise ValueError("Idea cannot be empty")
        return v.strip()
    
    @validator('channel_id')
    def validate_channel_id(cls, v):
        if not v.startswith('UC') and len(v) != 24:
            # Allow test channels for demo purposes
            if not v.startswith('test_') and v != 'demo_channel':
                raise ValueError("Invalid YouTube channel ID format")
        return v

class GeneratedTitle(BaseModel):
    """Individual generated title."""
    title: str = Field(..., description="Generated title text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Reasoning explanation")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality assessment score")
    pattern_match: List[str] = Field(default_factory=list, description="Matched patterns")
    similar_examples: List[str] = Field(default_factory=list, description="Similar high-performing titles")

class TitleGenerationResponse(BaseModel):
    """Response model for title generation."""
    titles: List[GeneratedTitle] = Field(..., description="Generated titles")
    channel_info: Dict[str, Any] = Field(default_factory=dict, description="Channel profile summary")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    request_id: str = Field(..., description="Unique request identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class ChannelAnalysisRequest(BaseModel):
    """Request model for channel analysis."""
    channel_id: str = Field(..., description="YouTube channel ID")
    refresh: bool = Field(default=False, description="Force refresh of cached data")

class ChannelAnalysisResponse(BaseModel):
    """Response model for channel analysis."""
    channel_id: str = Field(..., description="Channel ID")
    stats: Dict[str, Any] = Field(..., description="Channel statistics")
    patterns: Dict[str, Any] = Field(..., description="Title patterns")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    high_performers: List[Dict[str, Any]] = Field(..., description="Top performing videos")
    updated_at: datetime = Field(..., description="Last update timestamp")

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now)
    checks: Dict[str, Any] = Field(..., description="Individual health checks")
    uptime: float = Field(..., description="Uptime in seconds")

class MetricsResponse(BaseModel):
    """Metrics response model."""
    requests_total: int = Field(..., description="Total requests served")
    avg_response_time: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate percentage")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    active_connections: int = Field(..., description="Active connections")
    system_metrics: Dict[str, Any] = Field(..., description="System resource metrics")

# Rate limiting
class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self):
        self.requests = {}
        self.limits = {
            'default': (100, 3600),  # 100 requests per hour
            'premium': (1000, 3600),  # 1000 requests per hour for premium users
        }
    
    async def check_rate_limit(self, client_id: str, tier: str = 'default') -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        limit, window = self.limits.get(tier, self.limits['default'])
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < window
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= limit:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

# Global instances
rate_limiter = RateLimiter()
security = HTTPBearer(auto_error=False)

# Startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting TitleCraft AI API")
    
    # Initialize infrastructure
    cache_manager = await get_cache_manager()
    monitoring = get_monitoring_system()
    await monitoring.start_monitoring()
    
    # Initialize services
    app.state.start_time = time.time()
    app.state.data_store = DataStore()
    app.state.generation_engine = TitleGenerationEngine()
    
    # Load configuration and set up LLM orchestrator
    config = get_config()
    llm_configs = config.get('llm_providers', [])
    if llm_configs:
        app.state.llm_orchestrator = create_llm_orchestrator(llm_configs)
    else:
        logger.warning("No LLM providers configured")
        app.state.llm_orchestrator = None
    
    logger.info("TitleCraft AI API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down TitleCraft AI API")
    await monitoring.stop_monitoring()
    await shutdown_cache()
    logger.info("TitleCraft AI API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="TitleCraft AI API",
    description="Advanced YouTube title generation using AI and pattern analysis",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # Log request
    logger.info(
        "Request received",
        method=request.method,
        url=str(request.url),
        client_ip=client_ip,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    # Process request
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Track metrics
        monitoring = get_monitoring_system()
        monitoring.track_request(duration, success=response.status_code < 400)
        
        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration,
            client_ip=client_ip
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Track failed request
        monitoring = get_monitoring_system()
        monitoring.track_request(duration, success=False)
        
        logger.error(
            "Request failed",
            method=request.method,
            url=str(request.url),
            error=str(e),
            duration=duration,
            client_ip=client_ip
        )
        raise

# Dependencies
async def get_client_id(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract client ID for rate limiting."""
    if credentials:
        return f"auth_{credentials.credentials[:8]}"  # Use part of token as ID
    return request.client.host if request.client else "anonymous"

async def check_rate_limit(client_id: str = Depends(get_client_id)) -> str:
    """Check rate limits for requests."""
    if not await rate_limiter.check_rate_limit(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "3600"}
        )
    return client_id

# API Endpoints
@app.post(
    "/v1/generate_titles",
    response_model=TitleGenerationResponse,
    summary="Generate YouTube titles",
    description="Generate optimized YouTube titles based on idea and channel patterns"
)
async def generate_titles(
    request: TitleGenerationRequest,
    background_tasks: BackgroundTasks,
    client_id: str = Depends(check_rate_limit)
) -> TitleGenerationResponse:
    """Generate optimized YouTube titles."""
    start_time = time.time()
    request_id = f"req_{int(time.time())}_{hash(request.idea)}"
    
    logger.info(
        "Title generation requested",
        request_id=request_id,
        idea=request.idea[:50] + "..." if len(request.idea) > 50 else request.idea,
        channel_id=request.channel_id,
        client_id=client_id
    )
    
    try:
        # Check cache first
        cache_manager = await get_cache_manager()
        cache_key = f"{request.channel_id}:{hash(request.idea)}:{request.n_titles}"
        cached_result = await cache_manager.get('api_response', cache_key)
        
        if cached_result and not request.dict().get('force_refresh', False):
            logger.info("Returning cached result", request_id=request_id)
            processing_time = time.time() - start_time
            cached_result['processing_time'] = processing_time
            cached_result['request_id'] = request_id
            return TitleGenerationResponse(**cached_result)
        
        # Generate titles
        generation_engine = app.state.generation_engine
        
        # Convert request to internal format
        llm_request = LLMRequest(
            prompt=request.idea,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata={'idea': request.idea, 'channel_id': request.channel_id}
        )
        
        # Use generation engine
        result = await generation_engine.generate_titles_for_idea(
            idea=request.idea,
            channel_id=request.channel_id,
            n_titles=request.n_titles,
            include_reasoning=request.include_reasoning
        )
        
        processing_time = time.time() - start_time
        
        # Format response
        response_data = TitleGenerationResponse(
            titles=[
                GeneratedTitle(
                    title=title.get('title', ''),
                    confidence=title.get('confidence', 0.0),
                    reasoning=title.get('reasoning', '') if request.include_reasoning else None,
                    quality_score=title.get('quality_score', 0.0),
                    pattern_match=title.get('pattern_match', []),
                    similar_examples=title.get('similar_examples', [])
                ) for title in result.get('titles', [])
            ],
            channel_info=result.get('channel_info', {}),
            generation_metadata=result.get('metadata', {}),
            request_id=request_id,
            processing_time=processing_time
        )
        
        # Cache result
        background_tasks.add_task(
            cache_manager.set,
            'api_response',
            cache_key,
            response_data.dict(),
            ttl=3600  # 1 hour cache
        )
        
        logger.info(
            "Title generation completed",
            request_id=request_id,
            processing_time=processing_time,
            titles_generated=len(response_data.titles)
        )
        
        return response_data
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_handler = get_error_handler()
        
        error_result = await error_handler.handle_error(e, {
            'request_id': request_id,
            'idea': request.idea,
            'channel_id': request.channel_id
        })
        
        logger.error(
            "Title generation failed",
            request_id=request_id,
            error=str(e),
            processing_time=processing_time
        )
        
        # Return fallback response if available
        if error_result.get('fallback_data'):
            fallback_data = error_result['fallback_data']
            return TitleGenerationResponse(
                titles=[
                    GeneratedTitle(
                        title=title.get('title', 'Fallback Title'),
                        confidence=title.get('confidence', 0.1),
                        reasoning=title.get('reasoning', 'Fallback due to service error'),
                        quality_score=0.1,
                        pattern_match=[],
                        similar_examples=[]
                    ) for title in fallback_data.get('titles', [])
                ],
                request_id=request_id,
                processing_time=processing_time,
                generation_metadata={'fallback': True, 'error': str(e)}
            )
        
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post(
    "/v1/analyze_channel",
    response_model=ChannelAnalysisResponse,
    summary="Analyze channel patterns",
    description="Analyze channel performance patterns and provide optimization recommendations"
)
async def analyze_channel(
    request: ChannelAnalysisRequest,
    client_id: str = Depends(check_rate_limit)
) -> ChannelAnalysisResponse:
    """Analyze channel patterns and performance."""
    logger.info(f"Channel analysis requested for {request.channel_id}")
    
    try:
        # This would integrate with the actual channel analysis system
        # For now, return mock data
        return ChannelAnalysisResponse(
            channel_id=request.channel_id,
            stats={
                'total_videos': 150,
                'avg_views': 5000,
                'subscriber_count': 10000
            },
            patterns={
                'avg_title_length': 8.5,
                'question_ratio': 0.3,
                'numeric_ratio': 0.4
            },
            recommendations=[
                "Use more questions in titles",
                "Include numbers for better performance",
                "Keep titles between 6-10 words"
            ],
            high_performers=[
                {
                    'title': 'How to Learn Python in 30 Days',
                    'views': 50000,
                    'published': '2024-01-15'
                }
            ],
            updated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Channel analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check API and system health status"
)
async def health_check() -> HealthCheckResponse:
    """Get system health status."""
    try:
        monitoring = get_monitoring_system()
        health_data = await monitoring.get_health_status()
        uptime = time.time() - app.state.start_time
        
        return HealthCheckResponse(
            status=health_data['overall_health'],
            version="2.0.0",
            checks=health_data['checks'],
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            version="2.0.0",
            checks={"error": str(e)},
            uptime=0.0
        )

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics",
    description="Get detailed system performance metrics"
)
async def get_metrics(client_id: str = Depends(check_rate_limit)) -> MetricsResponse:
    """Get system performance metrics."""
    try:
        monitoring = get_monitoring_system()
        performance_data = monitoring.get_performance_summary(minutes=60)
        
        return MetricsResponse(
            requests_total=monitoring.total_requests,
            avg_response_time=performance_data.get('avg_response_time', 0.0),
            error_rate=performance_data.get('avg_error_rate', 0.0),
            cache_hit_rate=0.0,  # Would get from cache manager
            active_connections=1,  # Would get from connection pool
            system_metrics={
                'cpu_usage': performance_data.get('avg_cpu_usage', 0.0),
                'memory_usage': performance_data.get('avg_memory_usage', 0.0)
            }
        )
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

@app.get("/", include_in_schema=False)
async def root():
    """API root endpoint."""
    return {
        "message": "TitleCraft AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested endpoint {request.url.path} was not found",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "src.api.production_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )