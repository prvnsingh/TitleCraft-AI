# TitleCraft AI - Intelligent YouTube Title Generator

> **YouTube title generation powered by advanced AI with intelligent pattern discovery and multi-LLM support.**

An AI-powered system that analyzes successful YouTube title patterns and generates optimized, ranked titles for new video ideas using sophisticated machine learning techniques.

## 🚀 Key Features

- **🧠 Intelligent Pattern Discovery**: Dynamically analyzes channel performance data to identify successful title patterns
- **🎯 Context-Aware Generation**: Selects optimal prompts and parameters based on discovered channel characteristics  
- **📊 Quality-Driven Ranking**: Evaluates and ranks titles using pattern-based scoring with confidence metrics
- **⚡ Multi-LLM Support**: DeepSeek (default), OpenAI GPT-4, Anthropic Claude, HuggingFace models, and local Ollama
- **🛡️ Production Ready**: Comprehensive error handling, performance tracking, and graceful fallbacks
- **📈 Performance Analytics**: Real-time monitoring of response times, token usage, and cost tracking

## 🏗️ Architecture

TitleCraft AI implements a sophisticated 6-component pipeline for intelligent title generation:

![TitleCraft AI Architecture](Title%20craft.svg)

### Component Flow

```
TitleCraft FastAPI → Data Insights → Custom Prompt → LLM Service → Inference Evaluator → Response Builder
```

### 1. **TitleCraft FastAPI** 
*Entry point and orchestration layer*
- **FastAPI Server**: RESTful API with automatic OpenAPI documentation
- **Request Validation**: Pydantic models for type safety and validation
- **Health Monitoring**: System health checks and status endpoints
- **Error Handling**: Comprehensive exception handling with structured responses

### 2. **Data Insights**
*Intelligence extraction from historical performance data*
- **Pattern Discovery Agent**: Analyzes 211+ YouTube videos across channels to identify success patterns
- **Channel Analysis**: Classifies channels by volume (high/medium/low) and content type
- **Performance Correlation**: Calculates statistical correlations between title features and view counts
- **Adaptive Weighting**: Dynamically adjusts pattern importance based on predictive power

### 3. **Custom Prompt**
*Context-aware prompt engineering and strategy selection*
- **Context-Aware Prompt Selector**: Chooses optimal generation strategies based on channel characteristics
- **Dynamic Contextualization**: Incorporates channel-specific insights into LLM prompts
- **Strategy Optimization**: Different approaches for educational vs entertainment vs mixed content
- **Parameter Adaptation**: Automatically adjusts temperature, tokens, and other LLM parameters

### 4. **LLM Service**
*Unified multi-provider language model interface*
- **Multi-Provider Support**: OpenAI GPT-4, Anthropic Claude, HuggingFace (DeepSeek), Ollama local models
- **LangChain Integration**: Standardized interface across different LLM providers
- **Streaming Support**: Real-time response generation capabilities
- **Fallback Mechanisms**: Graceful degradation when primary models are unavailable

### 5. **Inference Evaluator**
*Quality assessment and performance prediction*
- **Title Quality Evaluator**: Scores titles against discovered success patterns
- **Pattern-Based Scoring**: Evaluates word count, questions, numbers, keywords, and formatting
- **Performance Prediction**: Predicts likely performance (high/medium/low) with confidence intervals
- **Confidence Scoring**: Provides reliability metrics for each generated title

### 6. **Response Builder**
*Final response assembly and ranking*
- **Intelligent Ranking**: Orders titles by predicted performance using weighted pattern analysis
- **Detailed Insights**: Provides reasoning, strengths, and recommendations for each title
- **Metadata Enrichment**: Adds performance metrics, token usage, and cost estimates
- **Structured Output**: Clean JSON responses with comprehensive title analytics

## 📋 Prerequisites

- **Python 3.8+**
- **API Keys** (optional but recommended):
  - OpenAI API key for GPT models
  - Anthropic API key for Claude models
  - HuggingFace token for additional models

## 🔧 Setup & Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd TitleCraft-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables (Optional)

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_key_here"  # Optional
$env:HUGGINGFACE_API_TOKEN="your_hf_token_here"   # Optional

# Linux/Mac
export OPENAI_API_KEY="your_openai_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_key_here"  # Optional
export HUGGINGFACE_API_TOKEN="your_hf_token_here"   # Optional
```

Or create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional for tracing
```

### 4. Run the Application

```bash
# Start the FastAPI server
python -m uvicorn src.apis.app:app --reload --port 8000
```

**✅ Server ready at**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs  
- **Health check**: http://localhost:8000/health

## 🚀 Quick Start Examples

### Generate Titles (Default DeepSeek Model)

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_id": "UC_example123",
    "idea": "How to learn Python programming",
    "n_titles": 4
  }'
```

### Generate with Specific Model

```bash
curl -X POST "http://localhost:8000/generate-with-model?model=openai-gpt4" \
  -H "Content-Type: application/json" \
  -d '{
    "channel_id": "UC_example123",
    "idea": "Best productivity tips for developers",
    "n_titles": 3
  }'
```

### Check System Health

```bash
curl http://localhost:8000/health
```

## 🤖 Supported LLM Providers

### **Default Model**
- **DeepSeek-R1-Distill-Qwen-32B**: Production default (HuggingFace) - Fast, efficient reasoning

### **Alternative Models**
- **OpenAI**: GPT-3.5-turbo, GPT-4 (fast and reliable)
- **Anthropic**: Claude-3 models (advanced reasoning)
- **HuggingFace**: Mistral, other open-source models
- **Ollama**: Local models (privacy-focused)

### **Key Capabilities**
- **Multiple Providers**: Switch between different LLM providers seamlessly
- **LangSmith Integration**: Optional tracing and debugging
- **Streaming Support**: Real-time response generation
- **Fallback Mechanisms**: Graceful degradation when APIs fail
- **Performance Tracking**: Monitor response times and costs

## 🔄 API Endpoints

### **Core Endpoints**
- `POST /generate` - Generate titles with DeepSeek (default)
- `POST /generate-with-model` - Generate titles with specific model
- `GET /models` - List all available models
- `GET /health` - Health check and system status

### **Interactive Documentation**
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

## 📊 Example Response

```json
{
  "titles": [
    {
      "title": "Master Python in 30 Days: Complete Beginner's Roadmap",
      "reasoning": "Strong alignment with successful patterns from this channel. Strong in: word_count, keyword_match, numeric_usage. High confidence in predictions due to robust channel data. [Performance: high, Score: 0.89, Confidence: 87%]",
      "confidence_score": 0.87,
      "model_used": "DeepSeek-R1-Distill-Qwen-32B"
    },
    {
      "title": "5 Python Projects That Will Land You a Job (Beginner to Pro)",
      "reasoning": "Good alignment with channel patterns, with room for optimization. Strong in: numeric_usage, keyword_match. Could improve: question_usage. [Performance: high, Score: 0.85, Confidence: 83%]",
      "confidence_score": 0.83,
      "model_used": "DeepSeek-R1-Distill-Qwen-32B"
    }
  ],
  "channel_id": "UC_example123",
  "idea": "How to learn Python programming",
  "request_id": "req_abc123",
  "model_used": "DeepSeek-R1-Distill-Qwen-32B",
  "provider": "huggingface",
  "response_time": 2.34,
  "tokens_used": 245,
  "estimated_cost": 0.0012,
  "success": true
}
```

## 💾 Data Analytics

The system analyzes **211 YouTube videos** from the training dataset:

### **Channel Analysis**
- Identifies top-performing title patterns (top 30% by views)
- Extracts successful title characteristics (length, keywords, formatting)
- Calculates performance correlations across different content types

### **Pattern Discovery**
- **Word Count Optimization**: Analyzes optimal title lengths
- **Question Usage**: Identifies effective question formats
- **Numeric Integration**: Discovers impact of numbers in titles
- **Keyword Matching**: Finds high-performing keywords per channel
- **Capitalization Patterns**: Analyzes formatting effectiveness

### **Available Test Channels**
- `UC510QYlOlKNyhy_zdQxnGYw` (Historical/Military content - 107 videos)
- Additional channels available in CSV training data

## 🚨 Fallback Behavior

If LLM APIs are unavailable, the system provides intelligent fallbacks:

- **Pattern-Based Generation**: Uses discovered successful patterns
- **Template-Based Titles**: Applies channel-specific templates
- **Graceful Degradation**: Ensures service availability without API keys
- **Detailed Logging**: Comprehensive error tracking and analytics

## 🛠️ Development

### **Running Tests**

```bash
# Run all tests
python -m pytest tests/

# Run specific test files
python tests/test_api.py
python tests/test_functionality.py
python tests/test_comprehensive.py
```

### **Code Structure**

```
📁 TitleCraft AI/
├── 📊 electrify__applied_ai_engineer__training_data.csv  # 211 videos, 3 channels
├── 📦 requirements.txt                                   # Core dependencies
├── 📖 README.md                                         # This documentation
├── 📂 src/
│   ├── 📂 apis/
│   │   ├── 🚀 app.py                                    # FastAPI application
│   │   ├── 📝 models.py                                 # Pydantic models
│   │   └── 🔧 __init__.py
│   ├── 📂 data_module/
│   │   ├── 📊 data_processor.py                         # CSV processing & analysis
│   │   └── 🔧 __init__.py
│   ├── 📂 services/
│   │   ├── 🧠 pattern_discovery.py                      # Intelligent pattern analysis
│   │   ├── 🎯 context_aware_prompts.py                  # Dynamic prompt selection
│   │   ├── 🤖 llm_service.py                           # Multi-provider LLM interface
│   │   ├── 📊 title_quality_evaluator.py               # Title scoring & ranking
│   │   ├── 🎬 title_generator.py                       # Main orchestration
│   │   ├── ⚙️ llm_config.py                            # Model configurations
│   │   ├── 📈 performance_tracker.py                   # Analytics & monitoring
│   │   ├── 📝 structured_logger.py                     # Logging system
│   │   └── 🔧 base_llm.py                              # Base LLM interface
│   └── 📂 logs/                                        # Application logs
└── 📂 docs/                                           # Documentation
    ├── 📖 LLM_SERVICE.md                               # LLM service details
    └── 📖 LOGGING_SYSTEM.md                           # Logging documentation
```

### **Adding New Models**

1. Update `llm_config.py` with model presets
2. Add provider configuration in `llm_service.py`
3. Update tests to include new model validation

### **Environment Configuration**

```bash
# Required for full functionality
OPENAI_API_KEY=your_openai_key_here

# Optional for enhanced capabilities
ANTHROPIC_API_KEY=your_anthropic_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=TitleCraft-AI
OLLAMA_BASE_URL=http://localhost:11434
```

## 🔒 Production Features

- **✅ Clean Architecture**: Well-structured, maintainable codebase
- **🚀 Efficient Default Model**: DeepSeek for fast, cost-effective generation
- **🔄 Alternative Models**: Premium options available when needed
- **🛡️ Comprehensive Error Handling**: Graceful failures and detailed logging
- **📊 Performance Tracking**: Real-time analytics and cost monitoring
- **🩺 Health Monitoring**: System status and service availability checks
- **📈 Scalable Design**: Ready for horizontal scaling and deployment

