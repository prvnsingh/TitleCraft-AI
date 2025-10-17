# TitleCraft AI - YouTube Title Generator# TitleCraft AI - AI-Powered YouTube Title Generator# TitleCraft AI - Minimal Take-Home Implementation



Production-ready YouTube title generation powered by DeepSeek AI with support for multiple LLM models.



## 🚀 Features**A streamlined agentic workflow that learns from high-performing YouTube titles and generates optimized titles for new video ideas.****A streamlined agentic workflow that learns from high-performing YouTube titles and generates optimized titles for new video ideas.**



- **DeepSeek Default**: Fast, efficient title generation with DeepSeek R1 model

- **Multi-Model Support**: Alternative models available via dedicated endpoint

- **Data-Driven**: Channel analysis for context-aware title generation## 🚀 Quick Start

- **Production Ready**: Clean API, error handling, and performance tracking

- **RESTful API**: Simple integration with comprehensive documentation## 🚀 Quick Start



## 📋 Requirements### Prerequisites



- Python 3.8+- Python 3.8+### Prerequisites

- FastAPI

- LangChain- OpenAI API key (required for OpenAI models)- Python 3.8+

- Transformers (HuggingFace)

- Optional: OpenAI, Anthropic API keys- Optional: Anthropic API key for Claude models- OpenAI API key (required for OpenAI models)



## 🔧 Installation- Optional: Anthropic API key, HuggingFace token for additional models



1. Clone the repository:### Setup & Run

```bash

git clone <repository-url>### Setup & Run

cd TitleCraft-AI

```1. **Install dependencies**



2. Create virtual environment:```bash1. **Install dependencies**

```bash

python -m venv venvpip install -r requirements.txt```bash

source venv/bin/activate  # On Windows: venv\Scripts\activate

``````pip install -r requirements.txt



3. Install dependencies:```

```bash

pip install -r requirements.txt2. **Set environment variables**

```

```bash2. **Set environment variables**

4. Set up environment variables (optional):

```bash# Windows PowerShell```bash

# Create .env file

OPENAI_API_KEY=your_openai_key$env:OPENAI_API_KEY="your_openai_api_key_here"# Windows PowerShell

ANTHROPIC_API_KEY=your_anthropic_key

HUGGINGFACE_API_TOKEN=your_hf_token$env:ANTHROPIC_API_KEY="your_anthropic_key_here"  # Optional$env:OPENAI_API_KEY="your_openai_api_key_here"

```

```

## 🎯 Quick Start

# OR set model-specific keys

### Start the API Server

```bash3. **Run the application**$env:GPT_3_5_TURBO_API_KEY="your_openai_key_here"

python -m uvicorn src.apis.app:app --host 0.0.0.0 --port 8000

``````bash$env:CLAUDE_3_SONNET_20240229_API_KEY="your_anthropic_key_here"



### Generate Titles (DeepSeek Default)# Start the FastAPI server```

```bash

curl -X POST "http://localhost:8000/generate" \python -m uvicorn src.apis.app:app --reload --port 8000

  -H "Content-Type: application/json" \

  -d '{```3. **Run the application**

    "channel_id": "UC_example123",

    "idea": "How to learn Python programming",```bash

    "n_titles": 4

  }'**✅ Server ready at**: http://localhost:8000# Start the Enhanced FastAPI server

```

- **Interactive docs**: http://localhost:8000/docs  python -m uvicorn src.apis.app:app --reload --port 8000

### Generate with Alternative Model

```bash- **Health check**: http://localhost:8000/health

curl -X POST "http://localhost:8000/generate-with-model?model=openai-gpt4" \

  -H "Content-Type: application/json" \# Check service status

  -d '{

    "channel_id": "UC_example123",## 🔧 Featurescurl http://localhost:8000/health

    "idea": "Best productivity tips for developers",

    "n_titles": 3```

  }'

```### Supported LLM Providers```bash



## 🔄 API Endpoints- **OpenAI**: GPT-3.5-turbo, GPT-4 (fast and reliable)# Windows



### Core Endpoints- **Anthropic**: Claude-3 models (advanced reasoning)  set OPENAI_API_KEY=your_api_key_here

- `POST /generate` - Generate titles with DeepSeek (default)

- `POST /generate-with-model` - Generate titles with specific model- **Ollama**: Local models (privacy-focused)

- `GET /models` - List all available models

- `GET /health` - Health check# Linux/Mac  



### Interactive Documentation### Key Capabilitiesexport OPENAI_API_KEY=your_api_key_here

- `GET /docs` - Swagger UI documentation

- `GET /redoc` - ReDoc documentation- **Multiple Models**: Switch between different LLM providers```



## 🤖 Available Models- **Performance Tracking**: Monitor response times and costs



### Default Model- **LangSmith Integration**: Optional tracing and debugging**✅ Server ready at**: http://localhost:8000

- `DeepSeek-R1-Distill-Qwen-32B`: Production default (HuggingFace)

- **Fallback Mechanisms**: Graceful degradation when APIs fail- **Interactive docs**: http://localhost:8000/docs  

### Alternative Models (via /generate-with-model)

- `openai-gpt4`: OpenAI GPT-4 (complex reasoning)- **Health check**: http://localhost:8000/health

- `anthropic-claude`: Anthropic Claude 3 Sonnet

- `hf-mistral`: Mistral 7B Instruct## 🏗️ Architecture- **LLM service status**: http://localhost:8000/llm/status



## 📊 Example Response



```json```## 🔧 Enhanced LLM Service

{

  "titles": [📁 TitleCraft AI/

    {

      "title": "Master Python in 30 Days: Complete Beginner's Roadmap",├── 📊 electrify__applied_ai_engineer__training_data.csv  # YouTube dataTitleCraft AI now includes a powerful LangChain-based LLM service with multiple provider support:

      "reasoning": "Uses time-bound promise and appeals to beginners seeking structured learning",

      "confidence_score": 0.92,├── 📦 requirements.txt                                   # Dependencies  

      "model_used": "DeepSeek-R1-Distill-Qwen-32B"

    }├── 📖 README.md                                         # Documentation### Supported Providers

  ],

  "channel_id": "UC_example123",├── 📂 src/- **OpenAI**: GPT-3.5-turbo, GPT-4 (fast and reliable)

  "idea": "How to learn Python programming", 

  "request_id": "req_abc123",│   ├── 📂 apis/- **Anthropic**: Claude-3 models (advanced reasoning)

  "model_used": "DeepSeek-R1-Distill-Qwen-32B",

  "provider": "huggingface",│   │   └── 🎯 app.py                                    # FastAPI application- **Ollama**: Local models (privacy-focused)

  "response_time": 2.34,

  "tokens_used": 245,│   ├── 📂 data_module/

  "estimated_cost": 0.0012,

  "success": true│   │   └── 📊 data_processor.py                         # Data analysis### Key Features

}

```│   └── 📂 services/- **LangSmith Tracing**: Monitor and debug LLM interactions



## 🛠 Configuration│       ├── 🤖 title_generator.py               # Main generator- **Plug-and-Play**: Switch between providers easily



### Model Configuration│       ├── ⚙️ llm_config.py                   # LLM configuration- **Streaming Support**: Real-time response generation

Edit `src/services/llm_config.py` to modify model settings or add new models.

│       ├── 🔧 llm_service.py                           # LLM service- **Fallback Mechanisms**: Graceful degradation

### Environment Setup

The system automatically uses available API keys from environment variables or `.env` file.│       ├── 📝 prompt_manager.py                        # Prompt templates



## 🔍 Data Processing│       └── 📊 performance_tracker.py                   # Performance monitoring### Quick Setup



The system analyzes YouTube channels to understand:└── 📂 tests/```bash

- Content patterns and themes

- Title structures and keywords      ├── 🧪 test_api.py                                  # API tests# Copy environment template

- Audience engagement indicators

- Channel-specific context    ├── 🧪 test_functionality.py                        # Functionality testscp .env.example .env



## 🚦 Error Handling    └── 🧪 test_comprehensive.py                        # Comprehensive tests



### Robust Fallbacks```# Edit .env with your API keys

- Automatic fallback title generation

- Graceful error responsesOPENAI_API_KEY=your_key_here

- Detailed error logging

- Pattern-based backup titles## 📋 API UsageANTHROPIC_API_KEY=your_key_here



## 🔒 Production FeaturesLANGCHAIN_API_KEY=your_langsmith_key_here  # Optional



- Clean, focused codebase### Generate Titles (Default Model)```

- Efficient default model (DeepSeek)

- Alternative models available when needed```bash

- Comprehensive error handling

- Performance trackingcurl -X POST "http://localhost:8000/generate" \See [LLM Service Documentation](docs/LLM_SERVICE.md) for detailed usage.

- Health monitoring

  -H "Content-Type: application/json" \

## 📝 Development

  -d '{

### Running the Server

```bash    "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",## 🏗️ Minimal Architecture

# Production mode

python -m uvicorn src.apis.app:app --host 0.0.0.0 --port 8000    "idea": "How to learn Python programming",



# Development mode with auto-reload    "n_titles": 4**Consolidated single-file implementation:**

python -m uvicorn src.apis.app:app --reload --host 0.0.0.0 --port 8000

```  }'```



### Testing```📁 TitleCraft AI/

```bash

python -m pytest tests/├── 📊 electrify_data.csv                    # Training data (211 videos, 3 channels)

```

### Generate Titles (Specific Model)├── 📦 requirements.txt                      # 7 core dependencies  

## 🤝 Contributing

```bash├── 📖 README.md                            # This documentation

1. Fork the repository

2. Create feature branch (`git checkout -b feature/AmazingFeature`)  curl -X POST "http://localhost:8000/generate-with-model?model=claude-fast" \└── 📂 src/api/

3. Commit changes (`git commit -m 'Add AmazingFeature'`)

4. Push to branch (`git push origin feature/AmazingFeature`)  -H "Content-Type: application/json" \    └── 🎯 production_app.py                # All-in-one implementation (15KB)

5. Open a Pull Request

  -d '{```

## 📄 License

    "channel_id": "UC_x5XG1OV2P6uZZ5FSM9Ttw",

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
    "idea": "Advanced machine learning techniques",**Data Flow:**

    "n_titles": 3,```

    "temperature": 0.8CSV Data → DataLoader → Channel Analysis → Title Generation → FastAPI Response

  }'                                          ↓

```                               Pattern-based Fallback (no API key needed)

```

### Get Available Models

```bash## � API Usage

curl "http://localhost:8000/models"

```### Generate Titles



### Performance Metrics**Endpoint:** `POST /generate`

```bash

curl "http://localhost:8000/performance?hours=24"**Request:**

``````json

{

## 🔬 Testing  "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",

  "idea": "How modern warfare tactics evolved from historical battles"

### Run All Tests}

```bash```

cd tests

python run_tests.py**Response (with OpenAI API key):**

``````json

{

### Run Specific Tests  "titles": [

```bash    {

# API tests      "title": "How Modern Military Tactics Were Born from Ancient Warfare", 

python tests/test_api.py      "reasoning": "Uses 'How' question format (found in 40% of top performers)...",

      "confidence": 0.85

# Functionality tests      }

python tests/test_functionality.py  ],

  "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",

# Comprehensive tests  "idea": "How modern warfare tactics evolved from historical battles"

python tests/test_comprehensive.py}

``````



## 📊 Data**Response (fallback without API key):**

```json

The application analyzes YouTube performance data from `electrify__applied_ai_engineer__training_data.csv` containing:{

- 211 video titles across 3 channels  "titles": [

- View counts and performance metrics    {

- Title patterns and optimization insights      "title": "How to modern warfare tactics evolved from historical battles",

      "reasoning": "Pattern-based title following channel's successful format (fallback generation)",  

## 🎯 Key Components      "confidence": 0.3

    },

### Data Processor    {

- Analyzes channel performance patterns      "title": "Why modern warfare tactics evolved from historical battles Works",

- Extracts successful title characteristics      "reasoning": "Pattern-based title following channel's successful format (fallback generation)",

- Provides data-driven insights for generation      "confidence": 0.3  

    }

### Enhanced Title Generator  ]

- Orchestrates the title generation workflow}

- Integrates multiple LLM providers```

- Tracks performance and provides fallbacks

## 🔧 Core Features

### LLM Service

- Unified interface for multiple providers- **✅ Verified Functionality**: All features tested and working

- LangChain-based implementation- **📊 Data Loading**: Loads 211 videos from 3 channels automatically

- Support for streaming and async operations- **🎯 Channel Analysis**: Identifies top-performing title patterns

- **🤖 Smart Generation**: OpenAI GPT-3.5 integration with fallback

### Performance Tracker- **🛡️ Error Handling**: Graceful degradation when API unavailable

- Monitors API response times- **📚 Auto Documentation**: FastAPI generates interactive API docs

- Tracks token usage and costs- **🔄 Pattern Matching**: Analyzes length, questions, numbers, keywords

- Provides detailed analytics

## � Testing the API

## 🚨 Fallback Behavior

### Using curl:

If LLM APIs are unavailable, the system provides pattern-based fallback titles:```bash

- Template-based generation using successful patternscurl -X POST "http://localhost:8000/generate" \

- No API key required for basic functionality     -H "Content-Type: application/json" \

- Graceful degradation ensures service availability     -d '{

       "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",

## 📝 Environment Variables       "idea": "The strategy behind famous military victories"

     }'

```bash```

# Required

OPENAI_API_KEY=your_openai_key### Using Python:

```python

# Optional  import requests

ANTHROPIC_API_KEY=your_anthropic_key

LANGCHAIN_API_KEY=your_langsmith_keyresponse = requests.post("http://localhost:8000/generate", json={

LANGCHAIN_PROJECT=TitleCraft-AI    "channel_id": "UC510QYlOlKNyhy_zdQxnGYw", 

OLLAMA_BASE_URL=http://localhost:11434    "idea": "The strategy behind famous military victories"

```})



## 🛠️ Developmentprint(response.json())

```

### Code Formatting

```bash## 🎬 Available Test Channels

# Format with black

python -m black src/ tests/ --line-length=88The training data includes 3 YouTube channels with 211 total videos:

```- `UC510QYlOlKNyhy_zdQxnGYw` (Historical/Military content - 107 videos)

- Additional channels available in CSV data

### Add New Models

1. Update `llm_config.py` with model presets## 🧪 Implementation Details

2. Add provider configuration in `llm_service.py`

3. Update tests to include new model### Minimalist Design Philosophy

- **Single File**: All functionality consolidated into `production_app.py` 

---- **No Over-Engineering**: Removed Redis, circuit breakers, complex monitoring

- **Focus on Requirements**: Core functionality without unnecessary complexity

**Built with FastAPI, LangChain, and modern Python practices for scalable AI title generation.**
### Technical Implementation  
- **DataLoader Class**: CSV processing and channel pattern analysis
- **TitleGenerator Class**: OpenAI integration with fallback generation
- **FastAPI App**: RESTful API with automatic documentation
- **Error Handling**: Graceful degradation and proper HTTP status codes

### Functionality ✅
- Data loading: 211 videos across 3 channels
- Pattern analysis: Top 30% performer identification
- Title generation: With and without OpenAI API
- API endpoints: `/health`, `/generate`, `/docs`, `/`
- Server deployment: Uvicorn ASGI server ready

---

