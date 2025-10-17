"""
LLM Service Implementation
Provides unified interface for multiple LLM providers using LangChain
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import os
import traceback

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.llms import Ollama
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage

from .base_llm import BaseLLM, LLMResponse, LLMModelConfig, LLMProvider
from .structured_logger import structured_logger

class LLMGenerationError(Exception):
    """Custom exception for LLM generation errors"""

    pass


@dataclass
class LLMServiceConfig:
    """Configuration for LLM Service"""

    default_temperature: float = 0.7
    default_max_tokens: int = 1500
    timeout: int = 60
    retry_attempts: int = 3


class LLMService(BaseLLM):
    """
    Concrete LLM Service implementation supporting multiple providers
    """

    def __init__(
        self, config: LLMModelConfig, service_config: Optional[LLMServiceConfig] = None
    ):
        """
        Initialize LLM service with model configuration

        Args:
            config: Model configuration
            service_config: Service-level configuration
        """
        self.service_config = service_config or LLMServiceConfig()
        self.logger = structured_logger
        super().__init__(config)

    def _initialize_llm(self):
        """Initialize the LLM instance based on provider"""
        return self._create_llm_instance()

    def _create_llm_instance(self):
        """Create the appropriate LLM instance based on provider"""
        if self.config.provider == LLMProvider.OPENAI:
            return self._create_openai_llm()
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return self._create_anthropic_llm()
        elif self.config.provider == LLMProvider.HUGGINGFACE:
            return self._create_huggingface_llm()
        elif self.config.provider == LLMProvider.OLLAMA:
            return self._create_ollama_llm()
        else:
            raise LLMGenerationError(f"Unsupported provider: {self.config.provider}")

    def _create_openai_llm(self):
        """Create OpenAI LLM instance"""
        api_key = self._get_api_key("OPENAI")
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=api_key,
            request_timeout=self.service_config.timeout,
        )

    def _create_anthropic_llm(self):
        """Create Anthropic LLM instance"""
        api_key = self._get_api_key("ANTHROPIC")
        return ChatAnthropic(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            anthropic_api_key=api_key,
            timeout=self.service_config.timeout,
        )

    def _create_huggingface_llm(self):
        """Create HuggingFace LLM instance"""
        try:
            api_key = self._get_api_key("HUGGINGFACE")
            self.logger.log_llm_operations({
                "event": "llm_initialization",
                "message": f"Creating HuggingFace LLM for model: {self.config.model_name}",
                "model": self.config.model_name,
                "provider": "huggingface",
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            })
            
            # Check if this is a conversational model that needs ChatHuggingFace
            is_conversational = self._is_conversational_model()
            
            # Try different approaches based on model type
            if is_conversational:
                self.logger.log_llm_operations({
                    "event": "llm_interface_creation",
                    "message": f"Creating conversational model interface for: {self.config.model_name}",
                    "model": self.config.model_name,
                    "interface_type": "conversational"
                })
                
                # For conversational models, we need to use the right task and let HF choose provider
                hf_endpoint = HuggingFaceEndpoint(
                    repo_id=self.config.model_name,
                    task="conversational",  # Use conversational task instead of text-generation
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_tokens,
                    huggingfacehub_api_token=api_key,
                    provider="auto",  # Let HuggingFace choose the best provider
                )
                
                # Wrap with ChatHuggingFace for proper chat interface
                self.logger.log_llm_operations({
                    "event": "llm_interface_success",
                    "message": f"Successfully created conversational HuggingFace endpoint for {self.config.model_name}",
                    "model": self.config.model_name,
                    "interface_type": "conversational"
                })
                return ChatHuggingFace(llm=hf_endpoint)
            else:
                self.logger.log_llm_operations({
                    "event": "llm_interface_creation",
                    "message": f"Creating text-generation model interface for: {self.config.model_name}",
                    "model": self.config.model_name,
                    "interface_type": "text-generation"
                })
                
                # For text generation models, use standard approach
                hf_endpoint = HuggingFaceEndpoint(
                    repo_id=self.config.model_name,
                    task="text-generation",
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_tokens,
                    huggingfacehub_api_token=api_key,
                    provider="auto",  # Let HuggingFace choose the best provider
                )
                
                self.logger.log_llm_operations({
                    "event": "llm_interface_success", 
                    "message": f"Successfully created text-generation HuggingFace endpoint for {self.config.model_name}",
                    "model": self.config.model_name,
                    "interface_type": "text-generation"
                })
                return hf_endpoint
            
        except Exception as e:
            self.logger.log_llm_operations({
                "event": "llm_creation_error",
                "message": f"Failed to create HuggingFace LLM: {str(e)}",
                "model": self.config.model_name,
                "error": str(e),
                "traceback": traceback.format_exc()
            }, level="ERROR")
            
            # If model fails due to task/provider mismatch, try alternative configurations
            if "not supported for task" in str(e) and "conversational" in str(e):
                self.logger.log_llm_operations({
                    "event": "llm_retry_attempt",
                    "message": f"Model {self.config.model_name} requires conversational task, retrying with ChatHuggingFace...",
                    "model": self.config.model_name,
                    "retry_strategy": "conversational_task"
                }, level="WARNING")
                try:
                    api_key = self._get_api_key("HUGGINGFACE")
                    hf_endpoint = HuggingFaceEndpoint(
                        repo_id=self.config.model_name,
                        task="conversational",
                        temperature=self.config.temperature,
                        max_new_tokens=self.config.max_tokens,
                        huggingfacehub_api_token=api_key,
                        provider="auto",
                    )
                    return ChatHuggingFace(llm=hf_endpoint)
                except Exception as retry_e:
                    self.logger.log_llm_operations({
                        "event": "llm_retry_failed",
                        "message": f"Retry with conversational task also failed: {str(retry_e)}",
                        "model": self.config.model_name,
                        "retry_error": str(retry_e)
                    }, level="ERROR")
            
            # Suggest working alternatives
            self.logger.log_llm_operations({
                "event": "llm_alternatives_suggested",
                "message": f"Model {self.config.model_name} failed to initialize. Consider these alternatives",
                "model": self.config.model_name,
                "alternatives": [
                    "google/flan-t5-large (text-generation)",
                    "microsoft/DialoGPT-medium (conversational)",
                    "gpt2 (text-generation)"
                ]
            }, level="WARNING")
            
            raise LLMGenerationError(
                f"Failed to initialize HuggingFace model {self.config.model_name}. "
                f"The model may not be available through the Inference API, requires special access, "
                f"or the provider doesn't support the required task. Error: {str(e)}"
            )

    def _create_ollama_llm(self):
        """Create Ollama LLM instance"""
        return Ollama(
            model=self.config.model_name,
            temperature=self.config.temperature,
        )

    def _get_api_key(self, provider: str) -> str:
        """
        Get API key for provider using modelId-APIKEY pattern

        Args:
            provider: Provider name (OPENAI, ANTHROPIC, etc.)

        Returns:
            API key string

        Raises:
            LLMGenerationError: If API key not found
        """
        # Try specific model-based key first: {MODEL_ID}_API_KEY
        model_key = f"{self.config.model_name.upper().replace('-', '_')}_API_KEY"
        api_key = os.getenv(model_key)

        if api_key:
            return api_key

        # Fallback to provider-based key: {PROVIDER}_API_KEY
        provider_key = f"{provider}_API_KEY"
        api_key = os.getenv(provider_key)

        if api_key:
            return api_key

        raise LLMGenerationError(
            f"API key not found. Set either {model_key} or {provider_key} environment variable"
        )

    def _filter_generation_kwargs(self, kwargs: dict) -> dict:
        """Filter and map generation parameters based on provider and model type"""
        generation_kwargs = {}
        
        if "temperature" in kwargs:
            generation_kwargs["temperature"] = kwargs["temperature"]
            
        if "max_tokens" in kwargs:
            # Map max_tokens to the appropriate parameter based on provider and model type
            if self.config.provider == LLMProvider.HUGGINGFACE:
                # Check if this is a conversational model that uses ChatHuggingFace
                is_conversational = self._is_conversational_model()
                
                if is_conversational:
                    # ChatHuggingFace uses max_tokens (not max_new_tokens)
                    generation_kwargs["max_tokens"] = kwargs["max_tokens"]
                else:
                    # Regular HuggingFaceEndpoint uses max_new_tokens
                    generation_kwargs["max_new_tokens"] = kwargs["max_tokens"]
            else:
                generation_kwargs["max_tokens"] = kwargs["max_tokens"]
        
        # For HuggingFace, filter out any unsupported parameters based on model type
        if self.config.provider == LLMProvider.HUGGINGFACE:
            is_conversational = self._is_conversational_model()
            
            if is_conversational:
                # ChatHuggingFace supported parameters
                allowed_params = {"temperature", "max_tokens", "top_p", "top_k"}
            else:
                # Regular HuggingFaceEndpoint supported parameters
                allowed_params = {"temperature", "max_new_tokens", "top_p", "top_k", "repetition_penalty"}
            
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if k in allowed_params}
        
        return generation_kwargs

    def _is_conversational_model(self) -> bool:
        """Check if the current model is a conversational model"""
        conversational_models = [
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large", 
            "facebook/blenderbot-400M-distill",
        ]
        
        model_name_lower = self.config.model_name.lower()
        return (
            any(conv_model.lower() == model_name_lower for conv_model in conversational_models) or
            "chat" in model_name_lower or
            "dialog" in model_name_lower or
            "conversation" in model_name_lower
        )

    def generate(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """
        Generate response using the configured LLM

        Args:
            messages: List of messages for the conversation
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse: Response with content and metadata
        """
        start_time = time.time()

        try:
            self.logger.log_llm_request(
                model=self.config.model_name,
                parameters={
                    "provider": self.config.provider.value,
                    "messages_count": len(messages),
                    "generation_kwargs": kwargs
                }
            )
            
            # Filter and map generation parameters
            generation_kwargs = self._filter_generation_kwargs(kwargs)

            # Generate response
            if generation_kwargs:
                response = self.llm.invoke(messages, **generation_kwargs)
            else:
                response = self.llm.invoke(messages)

            # Extract content based on response type
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            # Calculate response time
            response_time = time.time() - start_time

            # Estimate tokens (rough approximation)
            estimated_tokens = self._estimate_tokens(content)

            # Log successful response
            self.logger.log_llm_response(
                response_text=content[:500] + "..." if len(content) > 500 else content,
                tokens_used=estimated_tokens,
                model=self.config.model_name,
                response_time=response_time,
                cost=self._estimate_cost(estimated_tokens)
            )

            return LLMResponse(
                content=content,
                model_used=self.config.model_name,
                provider=self.config.provider.value,
                response_time=response_time,
                tokens_used=estimated_tokens,
                cost=self._estimate_cost(estimated_tokens),
                metadata={
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "timestamp": time.time(),
                },
            )

        except Exception as e:
            response_time = time.time() - start_time
            self.logger.log_llm_operations({
                "event": "llm_generation_error",
                "message": f"Generation failed for {self.config.model_name}",
                "model": self.config.model_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "response_time": response_time,
                "generation_kwargs": generation_kwargs if 'generation_kwargs' in locals() else 'Not set',
                "messages_count": len(messages),
                "traceback": traceback.format_exc()
            }, level="ERROR")
            
            raise LLMGenerationError(
                f"Generation failed for {self.config.model_name}: {str(e)}"
            ) from e

    def generate_stream(self, messages: List[BaseMessage], **kwargs):
        """
        Generate streaming response from LLM

        Args:
            messages: List of messages for the conversation
            **kwargs: Additional generation parameters

        Yields:
            Streaming response chunks
        """
        try:
            self.logger.log_llm_request(
                model=self.config.model_name,
                parameters={
                    "provider": self.config.provider.value,
                    "streaming": True,
                    "generation_kwargs": kwargs
                }
            )
            
            # Filter and map generation parameters
            generation_kwargs = self._filter_generation_kwargs(kwargs)

            # Generate streaming response
            if generation_kwargs:
                for chunk in self.llm.stream(messages, **generation_kwargs):
                    yield chunk
            else:
                for chunk in self.llm.stream(messages):
                    yield chunk

        except Exception as e:
            self.logger.log_llm_operations({
                "event": "llm_streaming_error",
                "message": f"Streaming generation failed for {self.config.model_name}",
                "model": self.config.model_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }, level="ERROR")
            
            raise LLMGenerationError(
                f"Streaming generation failed for {self.config.model_name}: {str(e)}"
            ) from e

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on provider and tokens"""
        # Simple cost estimation - can be made more sophisticated
        cost_per_1k = {
            LLMProvider.OPENAI: 0.002,
            LLMProvider.ANTHROPIC: 0.003,
            LLMProvider.HUGGINGFACE: 0.001,
            LLMProvider.OLLAMA: 0.0,  # Local model
        }

        rate = cost_per_1k.get(self.config.provider, 0.002)
        return (tokens / 1000) * rate


# Convenience functions for creating messages
def create_system_message(content: str) -> SystemMessage:
    """Create a system message"""
    return SystemMessage(content=content)


def create_human_message(content: str) -> HumanMessage:
    """Create a human message"""
    return HumanMessage(content=content)


def create_ai_message(content: str) -> AIMessage:
    """Create an AI message"""
    return AIMessage(content=content)
