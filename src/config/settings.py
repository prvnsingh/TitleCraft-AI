"""
TitleCraft AI Configuration Management
Handles environment-specific settings and API configurations
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import json

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    # Legacy direct API support (backward compatibility)
    provider: str = "openai"  # openai, anthropic, local
    model: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30
    retry_attempts: int = 3
    
    # LangChain configuration
    use_langchain: bool = True
    langchain_providers: Dict[str, Dict[str, Any]] = None
    default_langchain_provider: str = "openai_primary"
    
    def __post_init__(self):
        """Initialize default LangChain providers if not set."""
        if self.langchain_providers is None:
            self.langchain_providers = {
                "openai_primary": {
                    "provider": "openai",
                    "model_name": self.model,
                    "api_key": self.api_key if self.provider == "openai" else None,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout
                },
                "anthropic_backup": {
                    "provider": "anthropic", 
                    "model_name": "claude-3-haiku-20240307",
                    "api_key": None,  # Set via environment
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "timeout": self.timeout
                },
                "ollama_local": {
                    "provider": "ollama",
                    "model_name": "llama2",
                    "base_url": "http://localhost:11434",
                    "temperature": self.temperature,
                },
                "huggingface_fallback": {
                    "provider": "huggingface",
                    "model_name": "microsoft/DialoGPT-medium", 
                    "temperature": self.temperature,
                    "max_tokens": 150
                }
            }

@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    database: str = "titlecraft"
    username: str = "titlecraft_user"
    password: Optional[str] = None
    pool_size: int = 10

@dataclass
class RedisConfig:
    """Redis configuration for caching"""
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ttl: int = 3600  # Default cache TTL in seconds

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list = None
    rate_limit: int = 100  # requests per minute
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]

@dataclass
class DataConfig:
    """Data processing configuration"""
    data_directory: str = "data"
    embeddings_model: str = "all-MiniLM-L6-v2"
    max_similar_examples: int = 5
    min_examples_for_profile: int = 10
    cache_profiles: bool = True

class Config:
    """Main configuration class for TitleCraft AI"""
    
    def __init__(self, env: str = None):
        self.env = env or os.getenv("ENVIRONMENT", "development")
        self.project_root = Path(__file__).parent.parent.parent
        
        # Initialize configurations
        self.llm = LLMConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.data = DataConfig()
        
        # Load environment-specific settings
        self._load_environment_config()
        
        # Override with environment variables
        self._load_from_environment()
    
    def _load_environment_config(self):
        """Load configuration based on environment"""
        config_file = self.project_root / "config" / f"{self.env}.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                self._update_from_dict(config_data)
    
    def _load_from_environment(self):
        """Override configuration with environment variables"""
        env_mappings = {
            # LLM Configuration
            "OPENAI_API_KEY": ("llm", "api_key"),
            "ANTHROPIC_API_KEY": None,  # Handle separately for LangChain providers
            "LLM_PROVIDER": ("llm", "provider"),
            "LLM_MODEL": ("llm", "model"),
            "LLM_TEMPERATURE": ("llm", "temperature", float),
            "USE_LANGCHAIN": ("llm", "use_langchain", bool),
            "DEFAULT_LANGCHAIN_PROVIDER": ("llm", "default_langchain_provider"),
            
            # Database Configuration
            "DATABASE_URL": ("database", "url"),
            "DB_HOST": ("database", "host"),
            "DB_PORT": ("database", "port", int),
            "DB_NAME": ("database", "database"),
            "DB_USER": ("database", "username"),
            "DB_PASSWORD": ("database", "password"),
            
            # Redis Configuration
            "REDIS_URL": ("redis", "url"),
            "REDIS_HOST": ("redis", "host"),
            "REDIS_PORT": ("redis", "port", int),
            "REDIS_PASSWORD": ("redis", "password"),
            
            # API Configuration
            "API_HOST": ("api", "host"),
            "API_PORT": ("api", "port", int),
            "API_DEBUG": ("api", "debug", bool),
            
            # Data Configuration
            "DATA_DIRECTORY": ("data", "data_directory"),
            "EMBEDDINGS_MODEL": ("data", "embeddings_model"),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None and config_path is not None:
                self._set_nested_config(config_path, value)
        
        # Handle LangChain provider API keys
        self._update_langchain_api_keys()
    
    def _set_nested_config(self, config_path: tuple, value: str):
        """Set nested configuration value with type conversion"""
        section_name = config_path[0]
        attr_name = config_path[1]
        type_converter = config_path[2] if len(config_path) > 2 else str
        
        section = getattr(self, section_name)
        
        # Type conversion
        if type_converter == bool:
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif type_converter == int:
            value = int(value)
        elif type_converter == float:
            value = float(value)
        
        setattr(section, attr_name, value)
    
    def _update_langchain_api_keys(self):
        """Update LangChain provider configurations with API keys from environment"""
        if not self.llm.langchain_providers:
            return
        
        # Map environment variables to provider API keys
        api_key_mappings = {
            "OPENAI_API_KEY": ["openai_primary", "openai_secondary"],
            "ANTHROPIC_API_KEY": ["anthropic_backup", "anthropic_primary"], 
            "HUGGINGFACE_API_TOKEN": ["huggingface_fallback"]
        }
        
        for env_var, provider_names in api_key_mappings.items():
            api_key = os.getenv(env_var)
            if api_key:
                for provider_name in provider_names:
                    if provider_name in self.llm.langchain_providers:
                        self.llm.langchain_providers[provider_name]["api_key"] = api_key
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate LLM configuration
        if not self.llm.api_key and self.llm.provider in ["openai", "anthropic"]:
            errors.append(f"API key required for {self.llm.provider}")
        
        # Validate data directory
        data_path = Path(self.data.data_directory)
        if not data_path.exists():
            try:
                data_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create data directory: {e}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
    
    def get_data_path(self, filename: str = None) -> Path:
        """Get path to data file"""
        data_path = Path(self.data.data_directory)
        if filename:
            return data_path / filename
        return data_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "llm": self.llm.__dict__,
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "api": self.api.__dict__,
            "data": self.data.__dict__,
            "environment": self.env
        }
    
    def save_config(self, filename: str = None):
        """Save current configuration to file"""
        if filename is None:
            filename = f"{self.env}.json"
        
        config_path = self.project_root / "config" / filename
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove sensitive data before saving
        config_dict = self.to_dict()
        if "api_key" in config_dict["llm"]:
            config_dict["llm"]["api_key"] = "***REDACTED***"
        if "password" in config_dict["database"]:
            config_dict["database"]["password"] = "***REDACTED***"
        if "password" in config_dict["redis"]:
            config_dict["redis"]["password"] = "***REDACTED***"
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

# Global configuration instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload configuration from environment"""
    global config
    config = Config()
    return config