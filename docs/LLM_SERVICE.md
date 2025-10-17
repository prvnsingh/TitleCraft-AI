# LLM Service Documentation

## Overview

The TitleCraft AI LLM service provides a plug-and-play interface for different Language Learning Models (LLMs) using the LangChain framework with LangSmith tracing capabilities. This service allows you to easily switch between different LLM providers and monitor your LLM usage with advanced tracing.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic Claude, and Ollama (local)
- **LangChain Integration**: Standardized interface for all LLM interactions
- **LangSmith Tracing**: Monitor, debug, and optimize your LLM calls
- **Plug-and-Play Architecture**: Easy to switch between providers
- **Streaming Support**: Real-time response streaming
- **Fallback Mechanisms**: Graceful degradation when services are unavailable
- **Configuration Management**: Environment-based configuration

## Supported Providers

### 1. OpenAI
- **Models**: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview
- **Requirements**: `OPENAI_API_KEY` environment variable
- **Best for**: General-purpose text generation, fast responses

### 2. Anthropic Claude
- **Models**: claude-3-haiku, claude-3-sonnet, claude-3-opus
- **Requirements**: `ANTHROPIC_API_KEY` environment variable  
- **Best for**: Long-form content, analysis, reasoning

### 3. Ollama (Local)
- **Models**: llama2, codellama, mistral, and many others
- **Requirements**: Ollama server running locally
- **Best for**: Privacy-focused, offline usage, custom models

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**:
   Copy `.env.example` to `.env` and configure your API keys:
   ```bash
   cp .env.example .env
   ```

3. **Configure API Keys**:
   Edit `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   LANGCHAIN_API_KEY=your_langsmith_key_here
   ```

## LangSmith Setup (Optional)

LangSmith provides powerful tracing and monitoring for your LLM calls:

1. **Sign up**: Go to [LangSmith](https://smith.langchain.com) and create an account
2. **Get API Key**: Generate an API key from your dashboard
3. **Configure Environment**:
   ```env
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=TitleCraft-AI
   ```

## Usage Examples

### Basic Usage

```python
from src.services.title_generator import TitleGenerator

# Initialize with default provider (OpenAI)
generator = TitleGenerator()

# Generate titles
titles = generator.generate_titles("UC123", "How to learn Python programming")

for title in titles:
    print(f"Title: {title.title}")
    print(f"Reasoning: {title.reasoning}")
```

### Using Different Providers

```python
# Use Anthropic Claude
generator = EnhancedTitleGenerator(
    llm_provider="anthropic", 
    model="claude-3-sonnet-20240229"
)

# Use local Ollama
generator = EnhancedTitleGenerator(
    llm_provider="ollama", 
    model="llama2"
)
```

### Switching Providers at Runtime

```python
generator = EnhancedTitleGenerator()

# Switch to Anthropic
generator.switch_provider("anthropic", model="claude-3-sonnet-20240229")

# Switch to Ollama
generator.switch_provider("ollama", model="llama2")
```

### Streaming Responses

```python
for chunk in generator.generate_titles_streaming("UC123", "Python tutorial"):
    if chunk.get("complete"):
        titles = chunk.get("titles", [])
        print(f"Final titles: {len(titles)} generated")
    else:
        print(f"Streaming: {chunk.get('chunk', '')}", end="")
```

### Direct LLM Service Usage

```python
from src.services.llm import (
    LLMServiceFactory, 
    create_system_message, 
    create_human_message
)

# Create service
service = LLMServiceFactory.create_openai_service()

# Create messages
messages = [
    create_system_message("You are a YouTube title expert"),
    create_human_message("Create a catchy title for a Python tutorial")
]

# Generate response
response = service.generate(messages)
print(response)
```

## API Endpoints

### Generate Titles
```http
POST /generate
Content-Type: application/json

{
    "channel_id": "UC123",
    "idea": "How to learn Python programming"
}
```

**Response**:
```json
{
    "titles": [
        {
            "title": "Master Python Programming in 30 Days",
            "reasoning": "Uses successful pattern 'Master X in Y Days' from channel data"
        }
    ],
    "channel_id": "UC123",
    "idea": "How to learn Python programming",
    "llm_info": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "tracing_enabled": true
    }
}
```

### Check LLM Status
```http
GET /llm/status
```

**Response**:
```json
{
    "enhanced_mode": true,
    "service_info": {
        "provider": "openai",
        "model": "gpt-3.5-turbo",
        "tracing_enabled": true,
        "langchain_project": "TitleCraft-AI"
    },
    "config_status": {
        "available_providers": ["openai", "anthropic"],
        "api_keys": {
            "openai": true,
            "anthropic": true
        }
    }
}
```

### Switch Provider
```http
POST /llm/switch
Content-Type: application/json

{
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229"
}
```

## Configuration

### Default Configurations

The service comes with pre-configured settings for common use cases:

- **`openai_fast`**: Fast responses with GPT-3.5-turbo
- **`openai_smart`**: Higher quality with GPT-4
- **`anthropic_balanced`**: Balanced responses with Claude Sonnet
- **`ollama_local`**: Local inference with Llama2

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Claude |
| `LANGCHAIN_API_KEY` | LangSmith API key | Optional |
| `LANGCHAIN_PROJECT` | LangSmith project name | `TitleCraft-AI` |
| `LANGCHAIN_TRACING_V2` | Enable tracing | `false` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `LLM_DEBUG` | Debug mode | `false` |
| `LLM_STREAMING` | Enable streaming | `true` |
| `LLM_TIMEOUT` | Request timeout (seconds) | `120` |

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```
   ImportError: No module named 'langchain'
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

2. **API Key Errors**:
   ```
   ValueError: OpenAI API key is required
   ```
   **Solution**: Set the appropriate API key in your `.env` file

3. **Ollama Connection Error**:
   ```
   Connection refused to localhost:11434
   ```
   **Solution**: Start Ollama server with `ollama serve`

### Fallback Mode

If LangChain is not available, the service automatically falls back to the original OpenAI integration:

- No LangSmith tracing
- Limited to OpenAI only
- Reduced functionality
- Still maintains API compatibility

### Debug Mode

Enable debug mode to see detailed logs:

```env
LLM_DEBUG=true
```

This will show:
- Request/response details
- Timing information
- Provider switching events
- Error details

## Monitoring with LangSmith

Once configured, LangSmith provides:

1. **Trace Visualization**: See the flow of your LLM calls
2. **Performance Metrics**: Track latency, tokens, costs
3. **Error Monitoring**: Debug failed requests
4. **Usage Analytics**: Understand usage patterns
5. **Model Comparison**: Compare different providers/models

Access your traces at: https://smith.langchain.com

## Best Practices

1. **Provider Selection**:
   - Use OpenAI for speed and reliability
   - Use Anthropic for complex reasoning tasks
   - Use Ollama for privacy-sensitive applications

2. **Error Handling**:
   - Always implement fallback mechanisms
   - Handle API rate limits gracefully
   - Log errors for debugging

3. **Performance**:
   - Use streaming for long responses
   - Cache frequently used patterns
   - Monitor token usage and costs

4. **Security**:
   - Never commit API keys to version control
   - Use environment variables for configuration
   - Rotate API keys regularly

## Development

### Running Tests

```bash
# Test the LLM service
python tests/test_llm_service.py

# Run all tests
python -m pytest tests/
```

### Adding New Providers

1. Extend the `LLMProvider` enum
2. Add initialization logic in `LangChainLLMService`
3. Update configuration templates
4. Add tests for the new provider

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the configuration validation output
3. Check LangSmith traces for debugging
4. Refer to LangChain documentation for advanced usage

## License

This service is part of the TitleCraft AI project and follows the same licensing terms.