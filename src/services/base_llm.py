"""
Abstract Base LLM Class
Provides the foundation for all LLM implementations with LangChain and LangSmith integration
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import os

from langchain.schema import BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client


class LLMProvider(Enum):
    """Supported LLM providers"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


@dataclass
class LLMModelConfig:
    """Configuration for LLM models"""

    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1500
    timeout: int = 30
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class LLMResponse:
    """Response from LLM with metadata for tracking"""

    content: str
    model_used: str
    provider: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.
    Handles LangChain initialization, LangSmith tracing, and provides
    standardized interface for different LLM providers.
    """

    def __init__(self, config: LLMModelConfig):
        """
        Initialize the LLM with configuration

        Args:
            config: LLMModelConfig containing provider, model, and parameters
        """
        self.config = config
        self.langsmith_enabled = self._setup_langsmith()
        self.callback_manager = self._setup_callbacks()
        self.llm = self._initialize_llm()

    def _setup_langsmith(self) -> bool:
        """
        Setup LangSmith tracing if API key is available

        Returns:
            bool: True if LangSmith is enabled, False otherwise
        """
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
                "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
            )
            os.environ["LANGCHAIN_PROJECT"] = os.getenv(
                "LANGCHAIN_PROJECT", "TitleCraft-AI"
            )
            return True
        return False

    def _setup_callbacks(self) -> Optional[CallbackManager]:
        """
        Setup callback manager for tracing and logging

        Returns:
            Optional[CallbackManager]: Callback manager if tracing is enabled
        """
        callbacks = []

        # Add LangSmith tracer if enabled
        if self.langsmith_enabled:
            tracer = LangChainTracer()
            callbacks.append(tracer)

        return CallbackManager(callbacks) if callbacks else None

    @abstractmethod
    def _initialize_llm(self):
        """
        Initialize the specific LLM implementation
        Must be implemented by subclasses

        Returns:
            The initialized LLM instance
        """
        pass

    @abstractmethod
    def generate(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """
        Generate response from LLM

        Args:
            messages: List of LangChain messages
            **kwargs: Additional parameters for generation

        Returns:
            LLMResponse: Response with metadata
        """
        pass


    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model

        Returns:
            Dict containing model information
        """
        return {
            "provider": self.config.provider.value,
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "langsmith_enabled": self.langsmith_enabled,
            "langchain_project": os.getenv("LANGCHAIN_PROJECT", "TitleCraft-AI"),
        }

    def update_temperature(self, temperature: float):
        """Update model temperature"""
        self.config.temperature = temperature
        if hasattr(self.llm, "temperature"):
            self.llm.temperature = temperature

    def update_max_tokens(self, max_tokens: int):
        """Update model max tokens"""
        self.config.max_tokens = max_tokens
        if hasattr(self.llm, "max_tokens"):
            self.llm.max_tokens = max_tokens
