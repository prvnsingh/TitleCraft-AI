"""
TitleCraft AI Processing Module

Advanced processing components for intelligent title generation:
- Enhanced prompting with adaptive contexts
- Pattern profiling and analysis
- Semantic matching and similarity
- LLM orchestration and management
- LangChain integration adapters
"""

from .enhanced_prompting import EnhancedPromptEngineering, PromptTemplate, PromptContext
from .pattern_profiler import EnhancedPatternProfiler, StructuralPattern
from .semantic_matcher import SemanticMatcher, SimilarityMatch
from .llm_orchestrator import LLMOrchestrator, TitleGenerationRequest

# Optional LangChain integration (requires additional dependencies)
try:
    from .langchain_adapter import LangChainAdapter
    _langchain_available = True
except ImportError:
    LangChainAdapter = None
    _langchain_available = False

__all__ = [
    "EnhancedPromptEngineering",
    "PromptTemplate", 
    "PromptContext",
    "EnhancedPatternProfiler",
    "StructuralPattern",
    "SemanticMatcher",
    "SimilarityMatch", 
    "LLMOrchestrator",
    "TitleGenerationRequest",
]

if _langchain_available:
    __all__.append("LangChainAdapter")