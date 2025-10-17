"""
Enhanced LLM Configuration Manager
Environment-based model selection with comprehensive model configurations
"""

import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict

from .base_llm import LLMProvider, LLMModelConfig
from .llm_service import LLMService

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass


@dataclass
class ModelPreset:
    """Predefined model configuration preset"""

    name: str
    provider: LLMProvider
    model_name: str
    description: str
    use_case: str
    temperature: float = 0.7
    max_tokens: int = 1500
    timeout: int = 30
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

    def to_config(self) -> LLMModelConfig:
        """Convert preset to LLMModelConfig"""
        return LLMModelConfig(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            extra_params=self.extra_params,
        )


class LLMConfigManager:
    """
    Manages LLM configurations and provides unified access to models
    """

    # Model configurations - DeepSeek as default + essential alternatives
    MODEL_PRESETS = {
        # Default DeepSeek Model
        "DeepSeek-R1-Distill-Qwen-32B": ModelPreset(
            name="DeepSeek-R1-Distill-Qwen-32B",
            provider=LLMProvider.HUGGINGFACE,
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            description="DeepSeek R1 Distill Qwen 32B - Default production model",
            use_case="general",
            temperature=0.7,
            max_tokens=8000,
        ),
        # Alternative models for second API
        "openai-gpt4": ModelPreset(
            name="openai-gpt4",
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            description="OpenAI GPT-4 for complex reasoning",
            use_case="complex_reasoning",
            temperature=0.5,
            max_tokens=2000,
        ),
        "anthropic-claude": ModelPreset(
            name="anthropic-claude",
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            description="Anthropic Claude 3 Sonnet",
            use_case="general",
            temperature=0.5,
            max_tokens=2000,
        ),
        "hf-mistral": ModelPreset(
            name="hf-mistral",
            provider=LLMProvider.HUGGINGFACE,
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            description="Mistral 7B Instruct",
            use_case="instruction_following",
            temperature=0.6,
            max_tokens=1500,
        ),
    }

    def __init__(self, default_model: str = "DeepSeek-R1-Distill-Qwen-32B"):
        """Initialize the configuration manager with DeepSeek as default"""
        self.default_model = default_model
        self.custom_configs = self._load_custom_configs()

    def _load_custom_configs(self) -> Dict[str, ModelPreset]:
        """Load custom model configurations from file"""
        config_path = Path(__file__).parent.parent.parent / "custom_models.json"

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    custom_data = json.load(f)

                custom_presets = {}
                for name, data in custom_data.items():
                    provider = LLMProvider(data["provider"])
                    preset = ModelPreset(
                        name=name,
                        provider=provider,
                        model_name=data["model_name"],
                        description=data.get("description", "Custom model"),
                        use_case=data.get("use_case", "custom"),
                        temperature=data.get("temperature", 0.7),
                        max_tokens=data.get("max_tokens", 1500),
                        timeout=data.get("timeout", 30),
                        extra_params=data.get("extra_params", {}),
                    )
                    custom_presets[name] = preset

                return custom_presets
            except Exception as e:
                print(f"Warning: Failed to load custom configs: {e}")

        return {}

    def get_available_models(self) -> Dict[str, ModelPreset]:
        """Get all available model presets (built-in + custom)"""
        all_models = self.MODEL_PRESETS.copy()
        all_models.update(self.custom_configs)
        return all_models

    def get_model_config(self, model_name: Optional[str] = None) -> LLMModelConfig:
        """
        Get model configuration by name or default

        Args:
            model_name: Name of the model preset, if None uses default

        Returns:
            LLMModelConfig: Configuration for the specified model
        """
        if model_name is None:
            model_name = self.default_model

        # Check if model exists in presets
        all_models = self.get_available_models()
        if model_name not in all_models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(all_models.keys())}"
            )

        preset = all_models[model_name]
        return preset.to_config()

    def create_llm_service(self, model_name: Optional[str] = None) -> LLMService:
        """
        Create an LLM service instance

        Args:
            model_name: Name of the model preset, if None uses default

        Returns:
            LLMService: Configured LLM service
        """
        config = self.get_model_config(model_name)
        return LLMService(config)

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        all_models = self.get_available_models()
        if model_name not in all_models:
            raise ValueError(f"Model '{model_name}' not found")

        preset = all_models[model_name]
        return {
            "name": preset.name,
            "provider": preset.provider.value,
            "model_name": preset.model_name,
            "description": preset.description,
            "use_case": preset.use_case,
            "temperature": preset.temperature,
            "max_tokens": preset.max_tokens,
        }


# Global configuration manager instance
config_manager = LLMConfigManager()
