"""
LLM Orchestrator for TitleCraft AI
Manages LLM interactions, adaptive prompting, and API integration
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import time
import re
from datetime import datetime

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic  
except ImportError:
    anthropic = None

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    # Create dummy decorators if tenacity is not available
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def stop_after_attempt(*args):
        pass
    
    def wait_exponential(*args, **kwargs):
        pass
    
    def retry_if_exception_type(*args):
        pass

# LangChain integration
try:
    from .langchain_adapter import (
        LangChainLLMManager, 
        LLMConfig as LangChainLLMConfig,
        LLMProvider,
        LLMResponse,
        create_openai_config,
        create_anthropic_config,
        create_ollama_config,
        create_huggingface_config
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from ..utils import BaseComponent
from ..data.models import ChannelProfile, VideoData

@dataclass 
class TitleGenerationRequest:
    """Request structure for title generation"""
    idea: str
    channel_id: str
    n_titles: int = 4
    style_preferences: Optional[Dict[str, Any]] = None
    context: Optional[str] = None

@dataclass
class GeneratedTitle:
    """Individual generated title with metadata"""
    title: str
    reasoning: str
    confidence: float
    word_count: int
    pattern_matches: List[str]
    emotional_triggers: List[str]
    estimated_performance: Optional[float] = None

@dataclass 
class TitleGenerationResponse:
    """Response structure for title generation"""
    titles: List[GeneratedTitle]
    generation_metadata: Dict[str, Any]
    model_used: str
    processing_time: float
    request_id: str

class LLMOrchestrator(BaseComponent):
    """
    Manages LLM interactions with adaptive prompting and multiple provider support.
    
    Features:
    - Multi-provider support (OpenAI, Anthropic, Local)
    - Adaptive prompting based on channel patterns
    - Retry logic with exponential backoff
    - Response validation and parsing
    - Performance tracking
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize LLM systems
        self._initialize_clients()
        self._initialize_langchain()
        
        # Prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        # Performance tracking
        self.generation_stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'avg_response_time': 0.0,
            'model_usage': {}
        }
    
    def _initialize_clients(self):
        """Initialize LLM provider clients"""
        self.clients = {}
        
        # OpenAI client
        if self.config.llm.provider == "openai" and openai is not None:
            if not self.config.llm.api_key:
                self.logger.warning("OpenAI API key not configured")
            else:
                self.clients['openai'] = openai.OpenAI(
                    api_key=self.config.llm.api_key,
                    timeout=self.config.llm.timeout
                )
                self.logger.info("OpenAI client initialized")
        
        # Anthropic client
        if self.config.llm.provider == "anthropic" and anthropic is not None:
            if not self.config.llm.api_key:
                self.logger.warning("Anthropic API key not configured")
            else:
                self.clients['anthropic'] = anthropic.Anthropic(
                    api_key=self.config.llm.api_key,
                    timeout=self.config.llm.timeout
                )
                self.logger.info("Anthropic client initialized")
        
        if not self.clients:
            self.logger.warning("No LLM clients initialized - title generation will not work")
    
    def _initialize_langchain(self):
        """Initialize LangChain LLM manager if available and configured"""
        self.langchain_manager = None
        
        if not LANGCHAIN_AVAILABLE:
            self.logger.info("LangChain not available - using direct API integration only")
            return
        
        if not self.config.llm.use_langchain:
            logger.info("LangChain disabled in configuration")
            return
        
        try:
            self.langchain_manager = LangChainLLMManager()
            
            # Register providers from configuration
            for provider_name, provider_config in self.config.llm.langchain_providers.items():
                try:
                    # Create LangChain config from provider settings
                    provider_type = provider_config.get("provider", "openai")
                    
                    if provider_type == "openai":
                        lc_config = create_openai_config(
                            model=provider_config.get("model_name", "gpt-3.5-turbo"),
                            api_key=provider_config.get("api_key"),
                            temperature=provider_config.get("temperature", 0.7),
                            max_tokens=provider_config.get("max_tokens")
                        )
                    elif provider_type == "anthropic":
                        lc_config = create_anthropic_config(
                            model=provider_config.get("model_name", "claude-3-haiku-20240307"),
                            api_key=provider_config.get("api_key"),
                            temperature=provider_config.get("temperature", 0.7),
                            max_tokens=provider_config.get("max_tokens")
                        )
                    elif provider_type == "ollama":
                        lc_config = create_ollama_config(
                            model=provider_config.get("model_name", "llama2"),
                            base_url=provider_config.get("base_url", "http://localhost:11434"),
                            temperature=provider_config.get("temperature", 0.7)
                        )
                    elif provider_type == "huggingface":
                        lc_config = create_huggingface_config(
                            model=provider_config.get("model_name", "microsoft/DialoGPT-medium"),
                            temperature=provider_config.get("temperature", 0.7),
                            max_tokens=provider_config.get("max_tokens", 150)
                        )
                    else:
                        logger.warning(f"Unsupported LangChain provider type: {provider_type}")
                        continue
                    
                    # Register the adapter
                    self.langchain_manager.register_adapter(provider_name, lc_config)
                    logger.info(f"Registered LangChain adapter: {provider_name} ({provider_type})")
                    
                except Exception as e:
                    logger.error(f"Failed to register LangChain provider {provider_name}: {e}")
            
            # Set default provider
            if self.config.llm.default_langchain_provider in self.langchain_manager.list_adapters():
                logger.info(f"Using default LangChain provider: {self.config.llm.default_langchain_provider}")
            else:
                logger.warning(f"Default LangChain provider not available: {self.config.llm.default_langchain_provider}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LangChain manager: {e}")
            self.langchain_manager = None
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different scenarios"""
        
        base_template = """You are a YouTube title optimization expert specializing in {channel_type} content.

CHANNEL SUCCESS PROFILE:
{channel_profile}

PROVEN SUCCESSFUL EXAMPLES:
{similar_examples}

TASK: Create {n_titles} compelling titles for this video idea: "{idea}"

REQUIREMENTS:
1. Follow {channel_type} content style and proven patterns
2. Target ~{target_length} words (±2 words acceptable)
3. Include specific elements that drive performance for this channel
4. Each title must be unique and compelling
5. Provide clear reasoning citing specific examples and patterns

ANALYSIS OF SUCCESSFUL PATTERNS:
{pattern_analysis}

OUTPUT FORMAT (strict JSON):
[
  {{
    "title": "Generated Title Here",
    "reasoning": "This follows [specific pattern] seen in '[similar example title]' which achieved [views] views. Uses proven elements: [list elements].",
    "confidence": 0.85,
    "word_count": 8,
    "pattern_matches": ["pattern1", "pattern2"],
    "emotional_triggers": ["curiosity", "urgency"]
  }}
]

Generate titles that would realistically achieve high engagement for this channel:"""

        templates = {
            'default': base_template,
            'high_confidence': base_template + "\n\nFOCUS: Use only the highest-performing patterns from the analysis.",
            'creative_variant': base_template + "\n\nFOCUS: Create creative variations while maintaining proven elements.",
            'conservative': base_template + "\n\nFOCUS: Stay close to proven successful formulas with minimal deviation."
        }
        
        return templates
    
    async def generate_titles(self,
                            request: TitleGenerationRequest,
                            channel_profile: ChannelProfile,
                            similar_examples: List[VideoData]) -> TitleGenerationResponse:
        """
        Generate titles using LLM with adaptive prompting
        
        Args:
            request: Title generation request
            channel_profile: Channel-specific patterns and data
            similar_examples: Contextually similar high-performing videos
            
        Returns:
            TitleGenerationResponse with generated titles and metadata
        """
        start_time = time.time()
        request_id = self._generate_request_id()
        
        self.logger.info(f"Generating titles for request {request_id}")
        
        try:
            # Update stats
            self.generation_stats['total_requests'] += 1
            
            # Create adaptive prompt
            prompt = self._create_adaptive_prompt(
                request, channel_profile, similar_examples
            )
            
            # Call LLM with retry logic
            raw_response = await self._call_llm_with_retry(prompt)
            
            # Parse and validate response
            titles = self._parse_and_validate_response(raw_response, request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update stats
            self.generation_stats['successful_generations'] += 1
            self._update_response_time_stats(processing_time)
            
            # Create response
            response = TitleGenerationResponse(
                titles=titles,
                generation_metadata={
                    'prompt_length': len(prompt),
                    'channel_type': self._infer_channel_type(channel_profile),
                    'similar_examples_count': len(similar_examples),
                    'generation_strategy': self._determine_generation_strategy(channel_profile)
                },
                model_used=self.config.llm.model,
                processing_time=processing_time,
                request_id=request_id
            )
            
            self.logger.info(f"Successfully generated {len(titles)} titles in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.generation_stats['failed_generations'] += 1
            logger.error(f"Title generation failed for request {request_id}: {e}")
            
            # Return fallback response
            return self._create_fallback_response(request, str(e), request_id)
    
    def _create_adaptive_prompt(self,
                               request: TitleGenerationRequest,
                               channel_profile: ChannelProfile,
                               similar_examples: List[VideoData]) -> str:
        """Create context-aware prompt based on channel and examples"""
        
        # Determine channel type and strategy
        channel_type = self._infer_channel_type(channel_profile)
        generation_strategy = self._determine_generation_strategy(channel_profile)
        
        # Format channel profile summary
        profile_summary = self._format_channel_profile(channel_profile)
        
        # Format similar examples
        examples_text = self._format_similar_examples(similar_examples)
        
        # Analyze patterns for this specific request
        pattern_analysis = self._analyze_patterns_for_prompt(channel_profile, similar_examples)
        
        # Select appropriate template
        template = self.prompt_templates.get(generation_strategy, self.prompt_templates['default'])
        
        # Fill template
        prompt = template.format(
            channel_type=channel_type,
            channel_profile=profile_summary,
            similar_examples=examples_text,
            n_titles=request.n_titles,
            idea=request.idea,
            target_length=int(channel_profile.title_patterns.avg_length_words),
            pattern_analysis=pattern_analysis
        )
        
        return prompt
    
    def _format_channel_profile(self, profile: ChannelProfile) -> str:
        """Format channel profile for prompt"""
        stats = profile.stats
        patterns = profile.title_patterns
        
        return f"""- Total videos: {stats.total_videos}
- Average views: {stats.avg_views:,.0f}
- High performer threshold: {stats.high_performer_threshold:,.0f}
- Average title length: {patterns.avg_length_words:.1f} words
- Question format usage: {patterns.question_ratio:.0%}
- Numeric elements: {patterns.numeric_ratio:.0%}
- Superlative usage: {patterns.superlative_ratio:.0%}
- Emotional hooks: {patterns.emotional_hook_ratio:.0%}
- Channel type: {profile.channel_type}"""
    
    def _format_similar_examples(self, examples: List[VideoData]) -> str:
        """Format similar examples for prompt"""
        if not examples:
            return "No similar examples available."
        
        formatted = []
        for i, example in enumerate(examples[:5], 1):
            formatted.append(f"{i}. \"{example.title}\" - {example.views_in_period:,} views")
        
        return "\n".join(formatted)
    
    def _analyze_patterns_for_prompt(self, 
                                   profile: ChannelProfile,
                                   examples: List[VideoData]) -> str:
        """Analyze patterns specifically for this prompt"""
        
        analysis_parts = []
        
        # High-level success factors
        if hasattr(profile, 'success_factors') and profile.success_factors:
            factors = profile.success_factors
            if 'optimal_title_length' in factors:
                analysis_parts.append(f"Optimal length: {factors['optimal_title_length']}")
            
            if 'successful_patterns' in factors:
                patterns = factors['successful_patterns']
                high_usage = [k for k, v in patterns.items() if v > 0.3]
                if high_usage:
                    analysis_parts.append(f"High-usage patterns: {', '.join(high_usage)}")
        
        # Example analysis
        if examples:
            example_titles = [ex.title for ex in examples]
            
            # Common words in examples
            all_words = ' '.join(example_titles).lower().split()
            word_freq = {}
            for word in all_words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            if common_words:
                analysis_parts.append(f"Common successful words: {', '.join([w[0] for w in common_words])}")
            
            # Pattern detection in examples
            question_count = sum(1 for title in example_titles if self._is_question(title))
            if question_count > len(example_titles) * 0.4:
                analysis_parts.append("Question format shows strong performance")
            
            number_count = sum(1 for title in example_titles if re.search(r'\d', title))
            if number_count > len(example_titles) * 0.4:
                analysis_parts.append("Numeric elements show strong performance")
        
        return " | ".join(analysis_parts) if analysis_parts else "Limited pattern data available."
    
    # Note: Retry decorator disabled when tenacity not available
    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with exponential backoff retry - supports both LangChain and direct APIs"""
        
        # Try LangChain first if available and configured
        if self.langchain_manager and self.config.llm.use_langchain:
            try:
                response = await self._call_langchain_llm(prompt)
                return response
            except Exception as e:
                logger.warning(f"LangChain LLM call failed, falling back to direct API: {e}")
                # Fall through to direct API calls
        
        # Fallback to direct API calls
        provider = self.config.llm.provider
        model = self.config.llm.model
        
        if provider not in self.clients:
            raise ValueError(f"LLM provider '{provider}' not initialized")
        
        try:
            if provider == "openai":
                response = await self._call_openai(prompt, model)
            elif provider == "anthropic":
                response = await self._call_anthropic(prompt, model)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            # Track model usage
            self.generation_stats['model_usage'][model] = \
                self.generation_stats['model_usage'].get(model, 0) + 1
            
            return response
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    async def _call_langchain_llm(self, prompt: str) -> str:
        """Call LLM using LangChain adapter with automatic fallback"""
        
        # Try default provider first
        default_provider = self.config.llm.default_langchain_provider
        
        try:
            # Initialize adapter if needed
            await self.langchain_manager.initialize_adapter(default_provider)
            
            # Generate response
            response = await self.langchain_manager.generate_response(
                prompt=prompt,
                adapter_name=default_provider
            )
            
            # Track model usage
            model_key = f"{response.provider}_{response.model}"
            self.generation_stats['model_usage'][model_key] = \
                self.generation_stats['model_usage'].get(model_key, 0) + 1
            
            logger.debug(f"LangChain response from {response.provider} ({response.model}): {len(response.content)} chars")
            
            return response.content
            
        except Exception as e:
            logger.error(f"LangChain LLM call failed with {default_provider}: {e}")
            
            # Try fallback providers
            available_adapters = self.langchain_manager.list_adapters()
            fallback_providers = [name for name in available_adapters if name != default_provider]
            
            for fallback_provider in fallback_providers:
                try:
                    logger.info(f"Trying fallback provider: {fallback_provider}")
                    await self.langchain_manager.initialize_adapter(fallback_provider)
                    
                    response = await self.langchain_manager.generate_response(
                        prompt=prompt,
                        adapter_name=fallback_provider
                    )
                    
                    # Track model usage
                    model_key = f"{response.provider}_{response.model}"
                    self.generation_stats['model_usage'][model_key] = \
                        self.generation_stats['model_usage'].get(model_key, 0) + 1
                    
                    logger.info(f"Successful fallback to {response.provider} ({response.model})")
                    return response.content
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback provider {fallback_provider} also failed: {fallback_error}")
                    continue
            
            # If all providers failed, re-raise the original exception
            raise e
    
    async def _call_openai(self, prompt: str, model: str) -> str:
        """Call OpenAI API"""
        client = self.clients['openai']
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic(self, prompt: str, model: str) -> str:
        """Call Anthropic API"""
        client = self.clients['anthropic']
        
        response = await asyncio.to_thread(
            client.messages.create,
            model=model,
            max_tokens=self.config.llm.max_tokens,
            temperature=self.config.llm.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _parse_and_validate_response(self, 
                                   response: str,
                                   request: TitleGenerationRequest) -> List[GeneratedTitle]:
        """Parse and validate LLM JSON response"""
        
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")
            
            json_str = response[json_start:json_end]
            titles_data = json.loads(json_str)
            
            # Validate and convert to GeneratedTitle objects
            titles = []
            for i, title_data in enumerate(titles_data):
                try:
                    title = GeneratedTitle(
                        title=title_data.get('title', f'Generated Title {i+1}'),
                        reasoning=title_data.get('reasoning', 'No reasoning provided'),
                        confidence=float(title_data.get('confidence', 0.5)),
                        word_count=int(title_data.get('word_count', 
                                     len(title_data.get('title', '').split()))),
                        pattern_matches=title_data.get('pattern_matches', []),
                        emotional_triggers=title_data.get('emotional_triggers', [])
                    )
                    titles.append(title)
                except Exception as e:
                    logger.warning(f"Failed to parse title {i+1}: {e}")
                    # Create fallback title
                    titles.append(GeneratedTitle(
                        title=title_data.get('title', f'Fallback Title {i+1}'),
                        reasoning=f"Parsing error: {e}",
                        confidence=0.3,
                        word_count=len(title_data.get('title', '').split()),
                        pattern_matches=[],
                        emotional_triggers=[]
                    ))
            
            if not titles:
                raise ValueError("No valid titles generated")
            
            return titles
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            
            # Create fallback titles
            fallback_titles = []
            for i in range(request.n_titles):
                fallback_titles.append(GeneratedTitle(
                    title=f"Error: Could not generate title {i+1}",
                    reasoning=f"Response parsing failed: {str(e)}",
                    confidence=0.0,
                    word_count=0,
                    pattern_matches=[],
                    emotional_triggers=[]
                ))
            
            return fallback_titles
    
    def _create_fallback_response(self,
                                request: TitleGenerationRequest,
                                error_msg: str,
                                request_id: str) -> TitleGenerationResponse:
        """Create fallback response when generation fails"""
        
        fallback_titles = []
        for i in range(request.n_titles):
            fallback_titles.append(GeneratedTitle(
                title=f"Fallback Title {i+1} for: {request.idea[:30]}...",
                reasoning=f"Generation failed: {error_msg}",
                confidence=0.0,
                word_count=6,
                pattern_matches=[],
                emotional_triggers=[]
            ))
        
        return TitleGenerationResponse(
            titles=fallback_titles,
            generation_metadata={'error': error_msg, 'fallback': True},
            model_used='fallback',
            processing_time=0.0,
            request_id=request_id
        )
    
    # Helper methods
    def _infer_channel_type(self, profile: ChannelProfile) -> str:
        """Infer channel type from profile"""
        return getattr(profile, 'channel_type', 'general')
    
    def _determine_generation_strategy(self, profile: ChannelProfile) -> str:
        """Determine generation strategy based on profile"""
        if hasattr(profile, 'stats') and profile.stats.total_videos > 50:
            return 'high_confidence'
        elif hasattr(profile, 'stats') and profile.stats.total_videos < 10:
            return 'conservative'
        else:
            return 'default'
    
    def _is_question(self, title: str) -> bool:
        """Check if title is a question"""
        return ('?' in title or 
                title.lower().startswith(('why ', 'what ', 'how ', 'when ', 'where ', 'who ', 'which ')))
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        timestamp = int(time.time())
        return f"req_{timestamp}_{hash(str(timestamp)) % 10000:04d}"
    
    def _update_response_time_stats(self, response_time: float):
        """Update response time statistics"""
        current_avg = self.generation_stats['avg_response_time']
        total_requests = self.generation_stats['total_requests']
        
        # Calculate new average
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self.generation_stats['avg_response_time'] = new_avg
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.generation_stats.copy()
    
    def reset_stats(self):
        """Reset generation statistics"""
        self.generation_stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'avg_response_time': 0.0,
            'model_usage': {}
        }