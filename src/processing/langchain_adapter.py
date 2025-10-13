"""
LangChain Adapter for TitleCraft AI

This module provides a unified interface for different LLM providers using LangChain.
Supports OpenAI, Anthropic, HuggingFace, local models, and other LangChain-compatible providers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json

try:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.outputs import LLMResult
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.callbacks import BaseCallbackHandler
    
    # LLM Providers
    from langchain_openai import ChatOpenAI, OpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_community.llms import HuggingFacePipeline, Ollama
    from langchain_community.chat_models import ChatOllama
    from langchain_core.runnables import RunnablePassthrough
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain not available: {e}")
    LANGCHAIN_AVAILABLE = False
    BaseLanguageModel = object
    BaseCallbackHandler = object
    LLMResult = object
    HumanMessage = object
    SystemMessage = object
    AIMessage = object
    ChatPromptTemplate = object
    PromptTemplate = object

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class TitleCraftCallback(BaseCallbackHandler):
    """Custom callback handler for tracking LLM usage and performance."""
    
    def __init__(self):
        self.tokens_used = 0
        self.calls_made = 0
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        import time
        self.start_time = time.time()
        self.calls_made += 1
        logger.debug(f"LLM call started: {len(prompts)} prompts")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running."""
        import time
        self.end_time = time.time()
        
        # Extract token usage if available
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.tokens_used += token_usage.get('total_tokens', 0)
        
        duration = self.end_time - self.start_time if self.start_time else 0
        logger.debug(f"LLM call completed in {duration:.2f}s, tokens: {self.tokens_used}")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM encounters an error."""
        self.errors.append(str(error))
        logger.error(f"LLM error: {error}")


class BaseLLMAdapter(ABC):
    """Base class for LLM adapters."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.callback_handler = TitleCraftCallback()
        self._llm = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM instance."""
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model_name,
            "calls_made": self.callback_handler.calls_made,
            "tokens_used": self.callback_handler.tokens_used,
            "errors": len(self.callback_handler.errors),
        }


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI models."""
    
    async def initialize(self) -> None:
        """Initialize OpenAI LLM."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available for OpenAI adapter")
        
        params = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "callbacks": [self.callback_handler],
        }
        
        if self.config.api_key:
            params["openai_api_key"] = self.config.api_key
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.base_url:
            params["openai_api_base"] = self.config.base_url
        
        params.update(self.config.extra_params)
        
        self._llm = ChatOpenAI(**params)
        logger.info(f"Initialized OpenAI adapter with model: {self.config.model_name}")
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using OpenAI."""
        if not self._llm:
            await self.initialize()
        
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._llm.invoke, messages)
            
            return LLMResponse(
                content=response.content,
                provider=self.config.provider.value,
                model=self.config.model_name,
                tokens_used=self.callback_handler.tokens_used,
                metadata={"response_metadata": getattr(response, "response_metadata", {})}
            )
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic models."""
    
    async def initialize(self) -> None:
        """Initialize Anthropic LLM."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available for Anthropic adapter")
        
        params = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "callbacks": [self.callback_handler],
        }
        
        if self.config.api_key:
            params["anthropic_api_key"] = self.config.api_key
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        
        params.update(self.config.extra_params)
        
        self._llm = ChatAnthropic(**params)
        logger.info(f"Initialized Anthropic adapter with model: {self.config.model_name}")
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using Anthropic."""
        if not self._llm:
            await self.initialize()
        
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._llm.invoke, messages)
            
            return LLMResponse(
                content=response.content,
                provider=self.config.provider.value,
                model=self.config.model_name,
                tokens_used=self.callback_handler.tokens_used,
                metadata={"response_metadata": getattr(response, "response_metadata", {})}
            )
            
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for Ollama local models."""
    
    async def initialize(self) -> None:
        """Initialize Ollama LLM."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available for Ollama adapter")
        
        params = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "callbacks": [self.callback_handler],
        }
        
        if self.config.base_url:
            params["base_url"] = self.config.base_url
        
        params.update(self.config.extra_params)
        
        self._llm = ChatOllama(**params)
        logger.info(f"Initialized Ollama adapter with model: {self.config.model_name}")
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using Ollama."""
        if not self._llm:
            await self.initialize()
        
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._llm.invoke, messages)
            
            return LLMResponse(
                content=response.content,
                provider=self.config.provider.value,
                model=self.config.model_name,
                metadata={"response_metadata": getattr(response, "response_metadata", {})}
            )
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise


class HuggingFaceAdapter(BaseLLMAdapter):
    """Adapter for HuggingFace models."""
    
    async def initialize(self) -> None:
        """Initialize HuggingFace LLM."""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available for HuggingFace adapter")
        
        try:
            from transformers import pipeline
            
            # Create HuggingFace pipeline
            hf_pipeline = pipeline(
                "text-generation",
                model=self.config.model_name,
                temperature=self.config.temperature,
                max_new_tokens=self.config.max_tokens or 150,
                **self.config.extra_params
            )
            
            self._llm = HuggingFacePipeline(pipeline=hf_pipeline)
            logger.info(f"Initialized HuggingFace adapter with model: {self.config.model_name}")
            
        except ImportError:
            logger.error("transformers package required for HuggingFace adapter")
            raise
    
    async def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using HuggingFace."""
        if not self._llm:
            await self.initialize()
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._llm.invoke, full_prompt)
            
            return LLMResponse(
                content=response,
                provider=self.config.provider.value,
                model=self.config.model_name,
            )
            
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            raise


class LangChainLLMManager:
    """Main manager for LangChain LLM adapters."""
    
    def __init__(self):
        self.adapters: Dict[str, BaseLLMAdapter] = {}
        self.default_adapter: Optional[str] = None
    
    def register_adapter(self, name: str, config: LLMConfig) -> None:
        """Register a new LLM adapter."""
        adapter_map = {
            LLMProvider.OPENAI: OpenAIAdapter,
            LLMProvider.ANTHROPIC: AnthropicAdapter,
            LLMProvider.OLLAMA: OllamaAdapter,
            LLMProvider.HUGGINGFACE: HuggingFaceAdapter,
        }
        
        adapter_class = adapter_map.get(config.provider)
        if not adapter_class:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        self.adapters[name] = adapter_class(config)
        
        if not self.default_adapter:
            self.default_adapter = name
        
        logger.info(f"Registered {config.provider.value} adapter: {name}")
    
    async def initialize_adapter(self, name: str) -> None:
        """Initialize a specific adapter."""
        if name not in self.adapters:
            raise ValueError(f"Adapter not found: {name}")
        
        await self.adapters[name].initialize()
    
    async def initialize_all(self) -> None:
        """Initialize all registered adapters."""
        for name in self.adapters:
            try:
                await self.initialize_adapter(name)
            except Exception as e:
                logger.error(f"Failed to initialize adapter {name}: {e}")
    
    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        adapter_name: Optional[str] = None
    ) -> LLMResponse:
        """Generate response using specified or default adapter."""
        adapter_name = adapter_name or self.default_adapter
        
        if not adapter_name or adapter_name not in self.adapters:
            raise ValueError(f"No valid adapter available: {adapter_name}")
        
        adapter = self.adapters[adapter_name]
        return await adapter.generate_response(prompt, system_prompt)
    
    def get_adapter_stats(self, adapter_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for adapter(s)."""
        if adapter_name:
            return self.adapters[adapter_name].get_stats()
        
        return {name: adapter.get_stats() for name, adapter in self.adapters.items()}
    
    def list_adapters(self) -> List[str]:
        """List all registered adapters."""
        return list(self.adapters.keys())


# Convenience functions for easy setup
def create_openai_config(
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> LLMConfig:
    """Create OpenAI configuration."""
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model_name=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_anthropic_config(
    model: str = "claude-3-haiku-20240307",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> LLMConfig:
    """Create Anthropic configuration."""
    return LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )


def create_ollama_config(
    model: str = "llama2",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7
) -> LLMConfig:
    """Create Ollama configuration."""
    return LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name=model,
        base_url=base_url,
        temperature=temperature
    )


def create_huggingface_config(
    model: str = "microsoft/DialoGPT-medium",
    temperature: float = 0.7,
    max_tokens: int = 150
) -> LLMConfig:
    """Create HuggingFace configuration."""
    return LLMConfig(
        provider=LLMProvider.HUGGINGFACE,
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens
    )