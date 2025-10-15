"""
Configuration for LLM Service
Environment variables and default settings
"""
import os
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass

# LangSmith Configuration
LANGSMITH_CONFIG = {
    "api_key": os.getenv("LANGCHAIN_API_KEY"),
    "project": os.getenv("LANGCHAIN_PROJECT", "TitleCraft-AI"),
    "endpoint": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    "tracing_enabled": os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
}

# API Keys for different providers
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
}

# Default LLM configurations for different use cases
LLM_CONFIGS = {
    "title_generation": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1500,
        "description": "Optimized for creative title generation"
    },
    "content_analysis": {
        "provider": "openai", 
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 2000,
        "description": "Optimized for analytical content processing"
    },
    "fast_generation": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "temperature": 0.5,
        "max_tokens": 1000,
        "description": "Fast responses for quick tasks"
    },
    "local_ollama": {
        "provider": "ollama",
        "model": "llama2",
        "temperature": 0.7,
        "max_tokens": 1000,
        "description": "Local inference with Ollama"
    },
    "anthropic_claude": {
        "provider": "anthropic",
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.5,
        "max_tokens": 1500,
        "description": "Anthropic Claude for balanced responses"
    }
}

# Model mappings for different providers
MODEL_MAPPINGS = {
    "openai": {
        "fast": "gpt-3.5-turbo",
        "smart": "gpt-4",
        "latest": "gpt-4-turbo-preview"
    },
    "anthropic": {
        "fast": "claude-3-haiku-20240307",
        "balanced": "claude-3-sonnet-20240229",
        "smart": "claude-3-opus-20240229"
    },
    "ollama": {
        "small": "llama2:7b",
        "medium": "llama2:13b",
        "large": "llama2:70b",
        "code": "codellama",
        "chat": "llama2-chat"
    }
}

# Feature flags
FEATURES = {
    "streaming_enabled": os.getenv("LLM_STREAMING", "true").lower() == "true",
    "debug_mode": os.getenv("LLM_DEBUG", "false").lower() == "true",
    "cache_enabled": os.getenv("LLM_CACHE", "false").lower() == "true",
    "retry_enabled": os.getenv("LLM_RETRY", "true").lower() == "true",
    "timeout_seconds": int(os.getenv("LLM_TIMEOUT", "120"))
}

# Validation functions
def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are present"""
    validation = {}
    for provider, key in API_KEYS.items():
        if provider == "ollama_base_url":
            validation[provider] = bool(key)  # Always has default
        else:
            validation[provider] = bool(key)
    return validation

def get_available_providers() -> list:
    """Get list of providers with valid API keys"""
    validation = validate_api_keys()
    available = []
    
    if validation.get("openai"):
        available.append("openai")
    if validation.get("anthropic"):
        available.append("anthropic")
    # Ollama is always available (local)
    available.append("ollama")
    
    return available

def get_recommended_config(use_case: str = "title_generation") -> Dict[str, Any]:
    """Get recommended configuration for a specific use case"""
    if use_case in LLM_CONFIGS:
        config = LLM_CONFIGS[use_case].copy()
        
        # Check if the provider is available
        available_providers = get_available_providers()
        if config["provider"] not in available_providers:
            # Fallback to the first available provider
            if available_providers:
                config["provider"] = available_providers[0]
                if config["provider"] == "openai":
                    config["model"] = "gpt-3.5-turbo"
                elif config["provider"] == "anthropic":
                    config["model"] = "claude-3-sonnet-20240229"
                else:  # ollama
                    config["model"] = "llama2"
        
        return config
    else:
        # Return default configuration
        return LLM_CONFIGS["title_generation"]

# Environment setup helper
def setup_environment():
    """Setup environment variables for LangSmith tracing"""
    if LANGSMITH_CONFIG["api_key"] and LANGSMITH_CONFIG["tracing_enabled"]:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_CONFIG["api_key"]
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_CONFIG["project"]
        os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_CONFIG["endpoint"]
        return True
    return False

# Configuration validation
def validate_config() -> Dict[str, Any]:
    """Validate the current configuration and return status"""
    status = {
        "api_keys": validate_api_keys(),
        "available_providers": get_available_providers(),
        "langsmith_configured": bool(LANGSMITH_CONFIG["api_key"]),
        "features": FEATURES,
        "configs_available": list(LLM_CONFIGS.keys())
    }
    return status

if __name__ == "__main__":
    # Print current configuration status
    import json
    status = validate_config()
    print("LLM Service Configuration Status:")
    print(json.dumps(status, indent=2))