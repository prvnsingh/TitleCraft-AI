"""
Intelligent Title Generator

Orchestrates intelligent pattern discovery, context-aware generation, and quality evaluation
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
import uuid

from src.data_module.data_processor import DataLoader, GeneratedTitle
from .llm_config import config_manager
from .llm_service import create_system_message, create_human_message, LLMGenerationError
from .performance_tracker import performance_tracker
from .pattern_discovery import PatternDiscoveryAgent
from .context_aware_prompts import ContextAwarePromptSelector
from .title_quality_evaluator import TitleQualityEvaluator
from .structured_logger import structured_logger, log_llm_operation

@dataclass
class TitleGenerationRequest:
    """Request object for title generation"""

    channel_id: str
    video_idea: str
    n_titles: int = 4
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class TitleGenerationResponse:
    """Response object for title generation"""

    titles: List[GeneratedTitle]
    request_id: str
    model_used: str
    provider: str
    response_time: float
    tokens_used: Optional[int]
    estimated_cost: Optional[float]
    success: bool
    error_message: Optional[str] = None


class TitleGenerator:
    """Intelligent title generator with pattern discovery, context-aware generation, and quality evaluation"""

    def __init__(self, default_model: Optional[str] = None):
        """Initialize the intelligent title generator"""
        self.logger = structured_logger
        self.data_loader = DataLoader()
        self.default_model = default_model or "DeepSeek-R1-Distill-Qwen-32B"
        self.current_request_id = None  # Track current request for logging context
        
        # Initialize intelligent components
        self.pattern_agent = PatternDiscoveryAgent()
        self.prompt_selector = ContextAwarePromptSelector()
        self.quality_evaluator = TitleQualityEvaluator()
        
        self.logger.log_llm_operations({
            "event": "title_generator_initialized",
            "default_model": self.default_model,
            "components": ["pattern_agent", "prompt_selector", "quality_evaluator"]
        })

    @log_llm_operation("title_generation", "DeepSeek-R1-Distill-Qwen-32B")
    def generate_titles(self, request: TitleGenerationRequest) -> TitleGenerationResponse:
        """Generate titles for a video idea based on channel patterns"""
        start_time = time.time()
        self.current_request_id = str(uuid.uuid4())

        self.logger.info("Starting title generation process", extra={
            'extra_fields': {
                'component': 'title_generator',
                'action': 'generate_start',
                'video_idea': request.video_idea,
                'n_titles': request.n_titles,
                'model_name': request.model_name,
                'temperature': request.temperature,
                'max_tokens': request.max_tokens
            },
            'request_id': self.current_request_id,
            'channel_id': request.channel_id
        })

        try:
            response = self._generate_titles_success(request, start_time)
            
            self.logger.info("Title generation completed successfully", extra={
                'extra_fields': {
                    'component': 'title_generator',
                    'action': 'generate_success',
                    'titles_generated': len(response.titles),
                    'model_used': response.model_used,
                    'total_time': response.response_time,
                    'tokens_used': response.tokens_used,
                    'estimated_cost': response.estimated_cost
                },
                'request_id': self.current_request_id,
                'channel_id': request.channel_id
            })
            
            return response
            
        except Exception as e:
            self.logger.error("Title generation failed", extra={
                'extra_fields': {
                    'component': 'title_generator',
                    'action': 'generate_error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                'request_id': self.current_request_id,
                'channel_id': request.channel_id
            })
            
            return self._generate_titles_error(request, e, start_time)

    def _generate_titles_success(self, request: TitleGenerationRequest, start_time: float) -> TitleGenerationResponse:
        """Handle successful intelligent title generation"""
        model_name = request.model_name or self.default_model
        
        self.logger.info("Creating LLM service", extra={
            'extra_fields': {
                'component': 'title_generator',
                'action': 'llm_service_creation',
                'model_name': model_name
            },
            'request_id': self.current_request_id,
            'channel_id': request.channel_id
        })
        
        
        # Get intelligent patterns for context-aware generation
        self.logger.info("Loading channel data for pattern discovery", extra={
            'extra_fields': {
                'component': 'title_generator',
                'action': 'data_loading'
            },
            'request_id': self.current_request_id,
            'channel_id': request.channel_id
        })
        
        channel_videos = self.data_loader.get_channel_data(request.channel_id)
        
        self.logger.info("Channel data loaded", extra={
            'extra_fields': {
                'component': 'title_generator',
                'action': 'data_loaded',
                'videos_count': len(channel_videos)
            },
            'request_id': self.current_request_id,
            'channel_id': request.channel_id
        })
        
        intelligent_patterns = self.pattern_agent.discover_patterns(channel_videos)
        
        # Get context-aware prompt and parameters
        contextual_prompt = self.prompt_selector.select_optimal_context(intelligent_patterns)
        
        self.logger.info("Context-aware prompt selected", extra={
            'extra_fields': {
                'component': 'title_generator',
                'action': 'prompt_selection',
                'strategy': contextual_prompt.strategy.value,
                'temperature': contextual_prompt.parameters.temperature,
                'max_tokens': contextual_prompt.parameters.max_tokens
            },
            'request_id': self.current_request_id,
            'channel_id': request.channel_id
        })
        
        # Prepare messages and generation parameters
        messages = self._prepare_messages(request)
        generation_kwargs = self._prepare_intelligent_generation_kwargs(request, contextual_prompt)
        
        llm_service = config_manager.create_llm_service(model_name)
        # Log LLM request details
        self.logger.log_llm_request(
            model=model_name,
            parameters=generation_kwargs,
            request_id=self.current_request_id,
            channel_id=request.channel_id
        )
        
        # Generate and parse response
        llm_response = llm_service.generate(messages, **generation_kwargs)
        
        # Log LLM response
        self.logger.log_llm_response(
            response_text=llm_response.content,
            tokens_used=llm_response.tokens_used,
            cost=llm_response.cost,
            response_time=time.time() - start_time,
            request_id=self.current_request_id,
            channel_id=request.channel_id
        )
        
        generated_titles = self._parse_llm_response(llm_response.content)
        
        self.logger.info("Titles parsed from LLM response", extra={
            'extra_fields': {
                'component': 'title_generator',
                'action': 'title_parsing',
                'titles_parsed': len(generated_titles)
            },
            'request_id': self.current_request_id,
            'channel_id': request.channel_id
        })
        
        # Intelligent quality evaluation and ranking
        title_scores = self.quality_evaluator.evaluate_and_rank_titles(generated_titles, intelligent_patterns)
        
        self.logger.info("Quality evaluation completed", extra={
            'extra_fields': {
                'component': 'title_generator',
                'action': 'quality_evaluation',
                'scores_computed': len(title_scores),
                'top_score': title_scores[0].overall_score if title_scores else None,
                'avg_confidence': sum(score.confidence_score for score in title_scores) / len(title_scores) if title_scores else None
            },
            'request_id': self.current_request_id,
            'channel_id': request.channel_id
        })
        
        # Convert to GeneratedTitle objects with enhanced reasoning
        enhanced_titles = []
        for i, score in enumerate(title_scores[:request.n_titles]):
            # Preserve original reasoning and append performance metrics if reasoning exists
            if score.reasoning and score.reasoning.strip():
                enhanced_reasoning = f"{score.reasoning.strip()} | Performance: {score.predicted_performance}, Score: {score.overall_score:.2f}"
            else:
                enhanced_reasoning = f"Generated based on channel patterns | Performance: {score.predicted_performance}, Score: {score.overall_score:.2f}"
            
            enhanced_title = GeneratedTitle(
                title=score.title,
                reasoning=enhanced_reasoning,
                confidence_score=score.confidence_score,
                model_used=model_name
            )
            enhanced_titles.append(enhanced_title)
            
            # Log individual title details
            self.logger.info(f"Enhanced title {i+1}", extra={
                'extra_fields': {
                    'component': 'title_generator',
                    'action': 'title_enhancement',
                    'title_index': i+1,
                    'title': score.title,
                    'confidence': score.confidence_score,
                    'overall_score': score.overall_score,
                    'predicted_performance': score.predicted_performance
                },
                'request_id': self.current_request_id,
                'channel_id': request.channel_id
            })
        
        # Track performance
        response_time = time.time() - start_time
        request_id = performance_tracker.track_request(
            system_prompt=messages[0].content,
            user_prompt=messages[1].content,
            llm_response=llm_response,
            prompt_template=f"intelligent_{contextual_prompt.strategy.value}",
            additional_metadata={
                "channel_id": request.channel_id,
                "video_idea": request.video_idea,
                "n_titles": request.n_titles,
                "channel_type": intelligent_patterns.channel_type,
                "content_style": intelligent_patterns.content_style,
                "pattern_confidence": intelligent_patterns.confidence_score,
                "avg_title_score": sum(s.overall_score for s in title_scores[:request.n_titles]) / min(len(title_scores), request.n_titles) if min(len(title_scores), request.n_titles) > 0 else 0.0
            }
        )
        
        return TitleGenerationResponse(
            titles=enhanced_titles,
            request_id=request_id,
            model_used=model_name,
            provider=llm_service.config.provider.value,
            response_time=response_time,
            tokens_used=llm_response.tokens_used,
            estimated_cost=llm_response.cost,
            success=True,
        )

    def _generate_titles_error(self, request: TitleGenerationRequest, error: Exception, start_time: float) -> TitleGenerationResponse:
        """Handle title generation errors with fallback"""
        try:
            request_id = performance_tracker.track_error(
                system_prompt="",
                user_prompt="",
                error=error,
                model_name="unknown",
                provider="unknown",
                prompt_template="title_generation",
                additional_metadata={
                    "channel_id": request.channel_id,
                    "video_idea": request.video_idea,
                    "n_titles": request.n_titles
                }
            )
        except Exception:
            request_id = "fallback"

        fallback_titles = self._generate_fallback_titles(request.video_idea, request.n_titles)
        
        return TitleGenerationResponse(
            titles=fallback_titles,
            request_id=request_id,
            model_used="unknown",
            provider="fallback",
            response_time=time.time() - start_time,
            tokens_used=None,
            estimated_cost=None,
            success=False,
            error_message=str(error),
        )

    def _prepare_messages(self, request: TitleGenerationRequest):
        """Prepare intelligent context-aware messages for LLM generation"""
        # Get channel data and discover intelligent patterns
        channel_videos = self.data_loader.get_channel_data(request.channel_id)
        intelligent_patterns = self.pattern_agent.discover_patterns(channel_videos)
        
        # Select optimal prompt based on discovered patterns
        contextual_prompt = self.prompt_selector.select_optimal_context(intelligent_patterns)
        
        # Format channel analysis for prompt
        channel_analysis = self._format_intelligent_analysis(intelligent_patterns, channel_videos)
        
        # Format prompts with variables
        system_prompt = contextual_prompt.system_prompt
        user_prompt = contextual_prompt.user_prompt_template.format(
            channel_analysis=channel_analysis,
            video_idea=request.video_idea,
            n_titles=request.n_titles
        )
        
        # Log prompt construction with injected data
        self.logger.log_prompt_construction(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            data_injected={
                "channel_analysis": {
                    "channel_type": intelligent_patterns.channel_type,
                    "content_style": intelligent_patterns.content_style,
                    "confidence_score": intelligent_patterns.confidence_score,
                    "video_count": len(channel_videos),
                    "avg_views": intelligent_patterns.avg_views,
                    "pattern_weights": {
                        "word_count": intelligent_patterns.pattern_weights.word_count_weight,
                        "question": intelligent_patterns.pattern_weights.question_weight,
                        "numeric": intelligent_patterns.pattern_weights.numeric_weight
                    }
                },
                "video_idea": request.video_idea,
                "n_titles": request.n_titles
            },
            optimization_strategy=contextual_prompt.strategy.value,
            request_id=self.current_request_id,
            channel_id=request.channel_id
        )
        
        return [
            create_system_message(system_prompt),
            create_human_message(user_prompt),
        ]
    
    def _format_intelligent_analysis(self, patterns, channel_videos) -> str:
        """Format intelligent channel analysis for prompt"""
        
        analysis_parts = []
        
        # Basic channel info
        analysis_parts.append(f"Channel Type: {patterns.channel_type.replace('_', ' ').title()}")
        analysis_parts.append(f"Content Style: {patterns.content_style.replace('_', ' ').title()}")
        analysis_parts.append(f"Total Videos Analyzed: {len(channel_videos)}")
        
        if channel_videos:
            views = [v.views_in_period for v in channel_videos]
            analysis_parts.append(f"Average Views: {sum(views)/len(views):,.0f}")
            analysis_parts.append(f"Top Performer Views: {max(views):,.0f}")
        
        # Pattern insights
        analysis_parts.append(f"\nSUCCESS PATTERNS (Confidence: {patterns.confidence_score:.1%}):")
        analysis_parts.append(f"• Optimal Word Count: ~{patterns.avg_word_count:.0f} words")
        analysis_parts.append(f"• Question Usage: {patterns.question_percentage:.0%} effective rate")
        analysis_parts.append(f"• Number Usage: {patterns.numeric_percentage:.0%} effective rate")
        analysis_parts.append(f"• Exclamation Usage: {patterns.exclamation_percentage:.0%} effective rate")
        
        if patterns.top_keywords:
            analysis_parts.append(f"• High-Performing Keywords: {', '.join(patterns.top_keywords[:8])}")
        
        # Pattern weights (for transparency)
        weights = patterns.pattern_weights
        analysis_parts.append("\nPATTERN IMPORTANCE WEIGHTS:")
        analysis_parts.append(f"• Word Count: {weights.word_count_weight:.2f}")
        analysis_parts.append(f"• Questions: {weights.question_weight:.2f}")
        analysis_parts.append(f"• Numbers: {weights.numeric_weight:.2f}")
        analysis_parts.append(f"• Keywords: {weights.keyword_weight:.2f}")
        
        return "\n".join(analysis_parts)

    def _prepare_generation_kwargs(self, request: TitleGenerationRequest) -> dict:
        """Prepare basic generation parameters"""
        kwargs = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        return kwargs
    
    def _prepare_intelligent_generation_kwargs(self, request: TitleGenerationRequest, contextual_prompt) -> dict:
        """Prepare intelligent context-aware generation parameters"""
        # Use adaptive parameters from context-aware prompt selector
        adaptive_params = contextual_prompt.parameters
        
        kwargs = {
            "temperature": adaptive_params.temperature,
            "max_tokens": adaptive_params.max_tokens
        }
        
        # Override with user-specified parameters if provided
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
            
        # Add optional parameters if supported
        if adaptive_params.top_p is not None:
            kwargs["top_p"] = adaptive_params.top_p
        if adaptive_params.presence_penalty is not None:
            kwargs["presence_penalty"] = adaptive_params.presence_penalty
        
        return kwargs

    def _parse_llm_response(self, response_text: str) -> List[GeneratedTitle]:
        """Parse LLM response into GeneratedTitle objects"""
        # Clean the response text first
        cleaned_response = self._clean_response_text(response_text)
        
        try:
            return self._parse_structured_response(cleaned_response)
        except Exception:
            return self._simple_parse_response(cleaned_response)

    def _clean_response_text(self, response_text: str) -> str:
        """Clean response text by removing model internal reasoning"""
        import re
        
        # Remove <think>...</think> blocks (DeepSeek model internal reasoning)
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<reasoning>.*?</reasoning>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<internal>.*?</internal>', '', response_text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        response_text = re.sub(r'\n\s*\n', '\n\n', response_text)
        return response_text.strip()

    def _parse_structured_response(self, response_text: str) -> List[GeneratedTitle]:
        """Parse structured LLM response"""
        titles = []
        lines = response_text.strip().split("\n")
        current_title = None
        current_reasoning = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if self._is_title_line(line):
                if current_title:
                    titles.append(GeneratedTitle(title=current_title, reasoning=current_reasoning.strip()))
                current_title = self._extract_title_text(line)
                current_reasoning = ""
            elif self._is_reasoning_line(line):
                current_reasoning = self._extract_reasoning_text(line)
            elif current_title and self._is_continuation_line(line):
                current_reasoning += " " + line

        if current_title:
            titles.append(GeneratedTitle(title=current_title, reasoning=current_reasoning.strip()))

        return titles

    def _is_continuation_line(self, line: str) -> bool:
        """Check if line is continuation of reasoning"""
        return not self._is_title_line(line) and not line.startswith("---") and not line.startswith("**")

    def _is_title_line(self, line: str) -> bool:
        """Check if line contains a title"""
        return (line.startswith("TITLE") or 
                line.startswith("**TITLE") or 
                line.startswith("### Title") or
                line.startswith("Title ") or
                line.startswith("**Title") or
                "TITLE" in line.upper() and ":" in line)

    def _is_reasoning_line(self, line: str) -> bool:
        """Check if line contains reasoning"""
        return (line.startswith("REASONING") or 
                line.startswith("**REASONING") or 
                line.startswith("**Reasoning:") or
                line.startswith("Reasoning:") or
                "REASONING" in line.upper() and ":" in line)
    
    def _extract_title_text(self, line: str) -> str:
        """Extract title text from line"""
        # Handle different title formats
        if ":" in line:
            title_part = line.split(":", 1)
            if len(title_part) > 1:
                title_text = title_part[1].strip()
                # Clean up markdown formatting and quotes
                title_text = title_text.strip('"').strip("'")
                title_text = title_text.replace("**", "").strip()
                title_text = title_text.replace("*", "").strip()
                # Remove surrounding quotes if they exist
                if title_text.startswith('"') and title_text.endswith('"'):
                    title_text = title_text[1:-1]
                return title_text
        return ""

    def _extract_reasoning_text(self, line: str) -> str:
        """Extract reasoning text from line"""
        reasoning_part = line.split(":", 1)
        if len(reasoning_part) > 1:
            reasoning_text = reasoning_part[1].strip()
            reasoning_text = reasoning_text.replace("**", "").strip()
            return reasoning_text
        return ""

    def _simple_parse_response(self, response_text: str) -> List[GeneratedTitle]:
        """Simple fallback parsing for when structured parsing fails"""
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        titles = []

        for line in lines:
            if self._is_potential_title(line):
                clean_title = self._clean_title_line(line)
                if clean_title and len(clean_title) > 5:
                    titles.append(GeneratedTitle(title=clean_title, reasoning="Generated based on channel patterns"))

        # If no titles found, take first few lines as titles
        if not titles:
            for line in lines[:4]:
                if len(line) > 5:
                    titles.append(GeneratedTitle(title=line, reasoning="Generated using LLM"))

        return titles

    def _is_potential_title(self, line: str) -> bool:
        """Check if line could be a title"""
        markers = ["title", "•", "-", "1.", "2.", "3.", "4."]
        return any(marker in line.lower() for marker in markers)

    def _clean_title_line(self, line: str) -> str:
        """Clean title line to extract just the title"""
        clean_title = line
        markers = ["TITLE 1:", "TITLE 2:", "TITLE 3:", "TITLE 4:", "TITLE:", "•", "-"]
        for marker in markers:
            clean_title = clean_title.replace(marker, "").strip()
        return clean_title

    def _generate_fallback_titles(
        self, idea: str, n_titles: int
    ) -> List[GeneratedTitle]:
        """Generate simple fallback titles when LLM fails"""
        fallback_patterns = [
            f"How to {idea}",
            f"The Ultimate Guide to {idea}",
            f"Everything You Need to Know About {idea}",
            f"Amazing {idea} Tips That Actually Work",
            f"Why {idea} is Important",
            f"The Truth About {idea}",
            f"Incredible {idea} Facts",
            f"Master {idea} in Minutes",
        ]

        titles = []
        for i, pattern in enumerate(fallback_patterns[:n_titles]):
            titles.append(
                GeneratedTitle(
                    title=pattern,
                    reasoning="Fallback title generated using pattern templates",
                )
            )

        return titles

    def get_available_models(self) -> Dict[str, Any]:
        """Get available models from config manager"""
        return config_manager.get_available_models()


# For backward compatibility, maintain the old class name as an alias
EnhancedTitleGenerator = TitleGenerator