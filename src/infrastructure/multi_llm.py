"""
Multi-Model LLM Support for TitleCraft AI
Implements support for multiple LLM providers with intelligent fallback and load balancing.
"""
import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

from .circuit_breaker import circuit_breaker, CircuitBreakerConfig
from .monitoring import get_logger, track_performance

logger = get_logger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    GROQ = "groq"
    COHERE = "cohere"

@dataclass
class LLMResponse:
    """Standard LLM response format."""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int
    cost_estimate: float = 0.0
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

@dataclass
class LLMRequest:
    """Standard LLM request format."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    model: Optional[str] = None
    provider_preference: Optional[List[LLMProvider]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    api_key: str
    base_url: Optional[str] = None
    default_model: str = ""
    max_tokens_per_minute: int = 50000
    max_requests_per_minute: int = 100
    cost_per_token: float = 0.0
    priority: int = 1  # Lower numbers = higher priority
    enabled: bool = True
    timeout: float = 30.0
    retry_attempts: int = 3

class BaseLLMAdapter(ABC):
    """Base class for LLM provider adapters."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = None
        self._initialize_client()
        
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider's client."""
        pass
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using this provider."""
        pass
    
    @abstractmethod
    def estimate_cost(self, request: LLMRequest) -> float:
        """Estimate the cost for this request."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass

class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI LLM adapter."""
    
    def _initialize_client(self):
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            self.available_models = [
                "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", 
                "gpt-3.5-turbo-16k", "gpt-4-1106-preview"
            ]
        except ImportError:
            logger.warning("OpenAI library not available")
            self.client = None
            
    @circuit_breaker("openai_llm", CircuitBreakerConfig(timeout=30.0))
    @track_performance("llm_generation", {"provider": "openai"})
    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.client:
            return LLMResponse(
                content="", 
                provider=LLMProvider.OPENAI,
                model="",
                tokens_used=0,
                success=False,
                error="OpenAI client not initialized"
            )
            
        start_time = time.time()
        model = request.model or self.config.default_model or "gpt-3.5-turbo"
        
        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            duration = time.time() - start_time
            tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.OPENAI,
                model=model,
                tokens_used=tokens_used,
                cost_estimate=self.estimate_cost(request),
                duration=duration,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"OpenAI generation failed: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.OPENAI,
                model=model,
                tokens_used=0,
                duration=duration,
                success=False,
                error=str(e)
            )
    
    def estimate_cost(self, request: LLMRequest) -> float:
        # Rough estimation based on prompt length
        estimated_tokens = len(request.prompt.split()) * 1.3  # Rough estimation
        return estimated_tokens * self.config.cost_per_token
    
    def get_available_models(self) -> List[str]:
        return self.available_models

class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude LLM adapter."""
    
    def _initialize_client(self):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self.available_models = [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307",
                "claude-2.1"
            ]
        except ImportError:
            logger.warning("Anthropic library not available")
            self.client = None
            
    @circuit_breaker("anthropic_llm", CircuitBreakerConfig(timeout=45.0))
    @track_performance("llm_generation", {"provider": "anthropic"})
    async def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.client:
            return LLMResponse(
                content="",
                provider=LLMProvider.ANTHROPIC,
                model="",
                tokens_used=0,
                success=False,
                error="Anthropic client not initialized"
            )
            
        start_time = time.time()
        model = request.model or self.config.default_model or "claude-3-sonnet-20240229"
        
        try:
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"{request.system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                prompt = f"Human: {prompt}\n\nAssistant:"
                
            response = await self.client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            duration = time.time() - start_time
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return LLMResponse(
                content=response.content[0].text,
                provider=LLMProvider.ANTHROPIC,
                model=model,
                tokens_used=tokens_used,
                cost_estimate=self.estimate_cost(request),
                duration=duration,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Anthropic generation failed: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.ANTHROPIC,
                model=model,
                tokens_used=0,
                duration=duration,
                success=False,
                error=str(e)
            )
    
    def estimate_cost(self, request: LLMRequest) -> float:
        estimated_tokens = len(request.prompt.split()) * 1.3
        return estimated_tokens * self.config.cost_per_token
    
    def get_available_models(self) -> List[str]:
        return self.available_models

class OllamaAdapter(BaseLLMAdapter):
    """Ollama local LLM adapter."""
    
    def _initialize_client(self):
        try:
            import aiohttp
            self.session = None
            self.base_url = self.config.base_url or "http://localhost:11434"
            self.available_models = [
                "llama2", "codellama", "mistral", "mixtral",
                "neural-chat", "starling-lm", "vicuna", "orca-mini"
            ]
        except ImportError:
            logger.warning("aiohttp not available for Ollama")
            
    @circuit_breaker("ollama_llm", CircuitBreakerConfig(timeout=60.0))
    @track_performance("llm_generation", {"provider": "ollama"})
    async def generate(self, request: LLMRequest) -> LLMResponse:
        import aiohttp
        
        start_time = time.time()
        model = request.model or self.config.default_model or "llama2"
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
                
            prompt = request.prompt
            if request.system_prompt:
                prompt = f"System: {request.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
                
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens
                    }
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time
                    
                    return LLMResponse(
                        content=result.get("response", ""),
                        provider=LLMProvider.OLLAMA,
                        model=model,
                        tokens_used=0,  # Ollama doesn't return token counts
                        cost_estimate=0.0,  # Local model, no cost
                        duration=duration,
                        metadata=result.get("context", {})
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")
                    
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Ollama generation failed: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.OLLAMA,
                model=model,
                tokens_used=0,
                duration=duration,
                success=False,
                error=str(e)
            )
    
    def estimate_cost(self, request: LLMRequest) -> float:
        return 0.0  # Local models have no API cost
    
    def get_available_models(self) -> List[str]:
        return self.available_models

class HuggingFaceAdapter(BaseLLMAdapter):
    """HuggingFace Inference API adapter."""
    
    def _initialize_client(self):
        try:
            import aiohttp
            self.session = None
            self.base_url = self.config.base_url or "https://api-inference.huggingface.co/models"
            self.available_models = [
                "microsoft/DialoGPT-large",
                "facebook/blenderbot-400M-distill",
                "microsoft/DialoGPT-medium",
                "gpt2-xl"
            ]
        except ImportError:
            logger.warning("aiohttp not available for HuggingFace")
            
    @circuit_breaker("huggingface_llm", CircuitBreakerConfig(timeout=45.0))
    @track_performance("llm_generation", {"provider": "huggingface"})
    async def generate(self, request: LLMRequest) -> LLMResponse:
        import aiohttp
        
        start_time = time.time()
        model = request.model or self.config.default_model or "microsoft/DialoGPT-large"
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
                
            async with self.session.post(
                f"{self.base_url}/{model}",
                json={
                    "inputs": request.prompt,
                    "parameters": {
                        "temperature": request.temperature,
                        "max_new_tokens": request.max_tokens,
                        "return_full_text": False
                    }
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    duration = time.time() - start_time
                    
                    # Handle different response formats
                    if isinstance(result, list) and len(result) > 0:
                        content = result[0].get("generated_text", "")
                    else:
                        content = str(result)
                    
                    return LLMResponse(
                        content=content,
                        provider=LLMProvider.HUGGINGFACE,
                        model=model,
                        tokens_used=0,  # HF doesn't return token counts
                        cost_estimate=0.0,  # Free tier
                        duration=duration
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"HuggingFace API error: {response.status} - {error_text}")
                    
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"HuggingFace generation failed: {e}")
            return LLMResponse(
                content="",
                provider=LLMProvider.HUGGINGFACE,
                model=model,
                tokens_used=0,
                duration=duration,
                success=False,
                error=str(e)
            )
    
    def estimate_cost(self, request: LLMRequest) -> float:
        return 0.0  # Free tier
    
    def get_available_models(self) -> List[str]:
        return self.available_models

class MultiLLMOrchestrator:
    """
    Orchestrates multiple LLM providers with intelligent fallback and load balancing.
    """
    
    def __init__(self, configs: List[ProviderConfig]):
        self.adapters: Dict[LLMProvider, BaseLLMAdapter] = {}
        self.configs = {config.provider: config for config in configs}
        self.usage_stats: Dict[LLMProvider, Dict[str, Any]] = {}
        self._initialize_adapters()
        
    def _initialize_adapters(self):
        """Initialize all configured LLM adapters."""
        adapter_classes = {
            LLMProvider.OPENAI: OpenAIAdapter,
            LLMProvider.ANTHROPIC: AnthropicAdapter,
            LLMProvider.OLLAMA: OllamaAdapter,
            LLMProvider.HUGGINGFACE: HuggingFaceAdapter,
        }
        
        for provider, config in self.configs.items():
            if not config.enabled:
                continue
                
            adapter_class = adapter_classes.get(provider)
            if adapter_class:
                try:
                    self.adapters[provider] = adapter_class(config)
                    self.usage_stats[provider] = {
                        'requests': 0,
                        'successes': 0,
                        'failures': 0,
                        'total_tokens': 0,
                        'total_cost': 0.0,
                        'avg_duration': 0.0
                    }
                    logger.info(f"Initialized {provider.value} adapter")
                except Exception as e:
                    logger.error(f"Failed to initialize {provider.value} adapter: {e}")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate text using the best available provider with intelligent fallback.
        """
        # Determine provider order
        providers = self._get_provider_order(request)
        
        if not providers:
            return LLMResponse(
                content="",
                provider=LLMProvider.OPENAI,  # Default
                model="",
                tokens_used=0,
                success=False,
                error="No available LLM providers"
            )
        
        last_error = None
        
        for provider in providers:
            if provider not in self.adapters:
                continue
                
            adapter = self.adapters[provider]
            stats = self.usage_stats[provider]
            
            try:
                logger.info(f"Attempting generation with {provider.value}")
                stats['requests'] += 1
                
                response = await adapter.generate(request)
                
                if response.success:
                    # Update success stats
                    stats['successes'] += 1
                    stats['total_tokens'] += response.tokens_used
                    stats['total_cost'] += response.cost_estimate
                    stats['avg_duration'] = (
                        (stats['avg_duration'] * (stats['successes'] - 1) + response.duration) 
                        / stats['successes']
                    )
                    
                    logger.info(
                        f"Successfully generated with {provider.value}",
                        duration=response.duration,
                        tokens=response.tokens_used
                    )
                    return response
                else:
                    stats['failures'] += 1
                    last_error = response.error
                    logger.warning(f"{provider.value} generation failed: {response.error}")
                    
            except Exception as e:
                stats['failures'] += 1
                last_error = str(e)
                logger.error(f"Error with {provider.value}: {e}")
                
        # All providers failed, return fallback response
        return self._create_fallback_response(request, last_error)

    def _get_provider_order(self, request: LLMRequest) -> List[LLMProvider]:
        """
        Determine the order to try providers based on preferences and performance.
        """
        available_providers = list(self.adapters.keys())
        
        # Use explicit preference if provided
        if request.provider_preference:
            preferred = [p for p in request.provider_preference if p in available_providers]
            remaining = [p for p in available_providers if p not in preferred]
            return preferred + remaining
        
        # Sort by priority, then by success rate
        def provider_score(provider: LLMProvider) -> tuple:
            config = self.configs[provider]
            stats = self.usage_stats[provider]
            
            # Calculate success rate
            total_requests = stats['requests']
            success_rate = stats['successes'] / max(total_requests, 1)
            
            # Lower priority number = higher priority
            # Higher success rate = better
            return (config.priority, -success_rate)
        
        return sorted(available_providers, key=provider_score)

    def _create_fallback_response(self, request: LLMRequest, last_error: str) -> LLMResponse:
        """Create a fallback response when all providers fail."""
        # Extract key information from the request
        idea = request.metadata.get('idea', 'Unknown Topic')
        
        fallback_titles = [
            f"Amazing {idea} - Complete Guide",
            f"How to Master {idea} in 2024", 
            f"The Ultimate {idea} Tutorial",
            f"Everything You Need to Know About {idea}"
        ]
        
        fallback_content = json.dumps([
            {
                "title": title,
                "reasoning": "Fallback response due to LLM service unavailability",
                "confidence": 0.2,
                "fallback": True
            } for title in fallback_titles[:2]
        ])
        
        return LLMResponse(
            content=fallback_content,
            provider=LLMProvider.OPENAI,  # Default provider for fallback
            model="fallback",
            tokens_used=0,
            cost_estimate=0.0,
            success=True,  # Mark as success since we provided fallback
            metadata={"fallback": True, "reason": last_error}
        )

    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status and statistics for all providers."""
        status = {}
        
        for provider, adapter in self.adapters.items():
            stats = self.usage_stats[provider]
            config = self.configs[provider]
            
            status[provider.value] = {
                'enabled': config.enabled,
                'available_models': adapter.get_available_models(),
                'requests': stats['requests'],
                'successes': stats['successes'],
                'failures': stats['failures'],
                'success_rate': (stats['successes'] / max(stats['requests'], 1)) * 100,
                'total_tokens': stats['total_tokens'],
                'total_cost': stats['total_cost'],
                'avg_duration': stats['avg_duration'],
                'priority': config.priority
            }
        
        return status

    async def test_providers(self) -> Dict[str, bool]:
        """Test all providers with a simple request."""
        test_request = LLMRequest(
            prompt="Hello, please respond with 'Test successful'",
            max_tokens=50,
            temperature=0.1
        )
        
        results = {}
        for provider, adapter in self.adapters.items():
            try:
                response = await adapter.generate(test_request)
                results[provider.value] = response.success
            except Exception as e:
                logger.error(f"Provider {provider.value} test failed: {e}")
                results[provider.value] = False
        
        return results

# Factory function for creating orchestrator from config
def create_llm_orchestrator(provider_configs: List[Dict[str, Any]]) -> MultiLLMOrchestrator:
    """Create LLM orchestrator from configuration."""
    configs = []
    
    for config_dict in provider_configs:
        try:
            provider = LLMProvider(config_dict['provider'])
            config = ProviderConfig(
                provider=provider,
                api_key=config_dict['api_key'],
                base_url=config_dict.get('base_url'),
                default_model=config_dict.get('default_model', ''),
                max_tokens_per_minute=config_dict.get('max_tokens_per_minute', 50000),
                max_requests_per_minute=config_dict.get('max_requests_per_minute', 100),
                cost_per_token=config_dict.get('cost_per_token', 0.0),
                priority=config_dict.get('priority', 1),
                enabled=config_dict.get('enabled', True),
                timeout=config_dict.get('timeout', 30.0),
                retry_attempts=config_dict.get('retry_attempts', 3)
            )
            configs.append(config)
        except Exception as e:
            logger.error(f"Invalid provider config: {config_dict}, error: {e}")
    
    return MultiLLMOrchestrator(configs)