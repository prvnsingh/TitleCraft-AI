"""
LLM Service with LangChain Framework and LangSmith Tracing
Provides a plug-and-play interface for different LLM providers
"""
import os
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configure LangSmith tracing
def setup_langsmith_tracing():
    """Setup LangSmith tracing if API key is available"""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "TitleCraft-AI")
        return True
    return False


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMConfig:
    """Configuration for LLM providers"""
    
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        **kwargs
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs


class BaseLLMService(ABC):
    """Abstract base class for LLM services"""
    
    @abstractmethod
    def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[BaseMessage], **kwargs):
        """Generate streaming response from LLM"""
        pass


class LangChainLLMService(BaseLLMService):
    """LangChain-based LLM service with tracing capabilities"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = self._initialize_llm()
        self.tracing_enabled = setup_langsmith_tracing()
        self.callback_manager = self._setup_callbacks()
    
    def _initialize_llm(self):
        """Initialize the LLM based on provider configuration"""
        if self.config.provider == LLMProvider.OPENAI:
            return self._initialize_openai()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._initialize_anthropic()
        elif self.config.provider == LLMProvider.OLLAMA:
            return self._initialize_ollama()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def _initialize_openai(self):
        """Initialize OpenAI LLM"""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        return ChatOpenAI(
            model=self.config.model,
            openai_api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra_params
        )
    
    def _initialize_anthropic(self):
        """Initialize Anthropic LLM"""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        return ChatAnthropic(
            model=self.config.model,
            anthropic_api_key=api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **self.config.extra_params
        )
    
    def _initialize_ollama(self):
        """Initialize Ollama LLM"""
        base_url = self.config.base_url or "http://localhost:11434"
        
        return Ollama(
            model=self.config.model,
            base_url=base_url,
            temperature=self.config.temperature,
            **self.config.extra_params
        )
    
    def _setup_callbacks(self):
        """Setup callback manager for tracing and logging"""
        callbacks = []
        
        # Add LangSmith tracer if tracing is enabled
        if self.tracing_enabled:
            tracer = LangChainTracer()
            callbacks.append(tracer)
        
        # Add streaming callback for development
        if os.getenv("LLM_DEBUG", "false").lower() == "true":
            callbacks.append(StreamingStdOutCallbackHandler())
        
        return CallbackManager(callbacks) if callbacks else None
    
    def generate(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate response from LLM with tracing"""
        try:
            # Override config with runtime parameters
            runtime_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
            }
            
            # Update LLM configuration
            for key, value in runtime_config.items():
                if hasattr(self.llm, key):
                    setattr(self.llm, key, value)
            
            # Generate response with callbacks
            if self.callback_manager:
                response = self.llm.invoke(
                    messages, 
                    config={"callbacks": self.callback_manager}
                )
            else:
                response = self.llm.invoke(messages)
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def generate_stream(self, messages: List[BaseMessage], **kwargs):
        """Generate streaming response from LLM"""
        try:
            # Override config with runtime parameters
            runtime_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
            }
            
            # Update LLM configuration
            for key, value in runtime_config.items():
                if hasattr(self.llm, key):
                    setattr(self.llm, key, value)
            
            # Stream response
            if self.callback_manager:
                for chunk in self.llm.stream(
                    messages, 
                    config={"callbacks": self.callback_manager}
                ):
                    yield chunk.content if hasattr(chunk, 'content') else str(chunk)
            else:
                for chunk in self.llm.stream(messages):
                    yield chunk.content if hasattr(chunk, 'content') else str(chunk)
                    
        except Exception as e:
            raise Exception(f"LLM streaming failed: {str(e)}")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "tracing_enabled": self.tracing_enabled,
            "langchain_project": os.getenv("LANGCHAIN_PROJECT", "TitleCraft-AI")
        }


class LLMServiceFactory:
    """Factory class for creating LLM services"""
    
    @staticmethod
    def create_service(provider: Union[str, LLMProvider], model: str, **kwargs) -> LangChainLLMService:
        """Create an LLM service instance"""
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        config = LLMConfig(provider=provider, model=model, **kwargs)
        return LangChainLLMService(config)
    
    @staticmethod
    def create_openai_service(model: str = "gpt-3.5-turbo", **kwargs) -> LangChainLLMService:
        """Create OpenAI LLM service"""
        return LLMServiceFactory.create_service(LLMProvider.OPENAI, model, **kwargs)
    
    @staticmethod
    def create_anthropic_service(model: str = "claude-3-sonnet-20240229", **kwargs) -> LangChainLLMService:
        """Create Anthropic LLM service"""
        return LLMServiceFactory.create_service(LLMProvider.ANTHROPIC, model, **kwargs)
    
    @staticmethod
    def create_ollama_service(model: str = "llama2", **kwargs) -> LangChainLLMService:
        """Create Ollama LLM service"""
        return LLMServiceFactory.create_service(LLMProvider.OLLAMA, model, **kwargs)


# Convenience functions for message creation
def create_system_message(content: str) -> SystemMessage:
    """Create a system message"""
    return SystemMessage(content=content)


def create_human_message(content: str) -> HumanMessage:
    """Create a human message"""
    return HumanMessage(content=content)


def create_ai_message(content: str) -> AIMessage:
    """Create an AI message"""
    return AIMessage(content=content)


# Default configurations for common use cases
DEFAULT_CONFIGS = {
    "openai_fast": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "openai_smart": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 2000
    },
    "anthropic_balanced": {
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.5,
        "max_tokens": 1500
    },
    "ollama_local": {
        "provider": LLMProvider.OLLAMA,
        "model": "llama2",
        "temperature": 0.7,
        "max_tokens": 1000
    }
}


def get_default_service(config_name: str = "openai_fast", **overrides) -> LangChainLLMService:
    """Get a pre-configured LLM service"""
    if config_name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(DEFAULT_CONFIGS.keys())}")
    
    config_dict = DEFAULT_CONFIGS[config_name].copy()
    config_dict.update(overrides)
    
    provider = config_dict.pop("provider")
    config = LLMConfig(provider=provider, **config_dict)
    return LangChainLLMService(config)


# Example usage and testing
if __name__ == "__main__":
    # Example: Create an OpenAI service
    llm_service = LLMServiceFactory.create_openai_service()
    
    # Example conversation
    messages = [
        create_system_message("You are a helpful assistant."),
        create_human_message("What is the capital of France?")
    ]
    
    # Generate response
    response = llm_service.generate(messages)
    print(f"Response: {response}")
    
    # Get provider info
    info = llm_service.get_provider_info()
    print(f"Provider Info: {info}")