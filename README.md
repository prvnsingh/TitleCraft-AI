# TitleCraft AI - Minimal Take-Home Implementation

**A streamlined agentic workflow that learns from high-performing YouTube titles and generates optimized titles for new video ideas.**


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (optional - works with fallback generation)

### Setup & Run

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the application**
```bash
# Direct server start
python -m uvicorn src.api.production_app:app --port 8000

# Or using the app directly
python -m src.api.production_app
```

3. **Set OpenAI API Key (Optional)**
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac  
export OPENAI_API_KEY=your_api_key_here
```

**✅ Server ready at**: http://localhost:8000
- **Interactive docs**: http://localhost:8000/docs  
- **Health check**: http://localhost:8000/health
- **LLM service status**: http://localhost:8000/llm/status

## 🔧 Enhanced LLM Service

TitleCraft AI now includes a powerful LangChain-based LLM service with multiple provider support:

### Supported Providers
- **OpenAI**: GPT-3.5-turbo, GPT-4 (fast and reliable)
- **Anthropic**: Claude-3 models (advanced reasoning)
- **Ollama**: Local models (privacy-focused)

### Key Features
- **LangSmith Tracing**: Monitor and debug LLM interactions
- **Plug-and-Play**: Switch between providers easily
- **Streaming Support**: Real-time response generation
- **Fallback Mechanisms**: Graceful degradation

### Quick Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here  # Optional
```

See [LLM Service Documentation](docs/LLM_SERVICE.md) for detailed usage.


## 🏗️ Minimal Architecture

**Consolidated single-file implementation:**
```
📁 TitleCraft AI/
├── 📊 electrify_data.csv                    # Training data (211 videos, 3 channels)
├── 📦 requirements.txt                      # 7 core dependencies  
├── 📖 README.md                            # This documentation
└── 📂 src/api/
    └── 🎯 production_app.py                # All-in-one implementation (15KB)
```

**Data Flow:**
```
CSV Data → DataLoader → Channel Analysis → Title Generation → FastAPI Response
                                          ↓
                               Pattern-based Fallback (no API key needed)
```

## � API Usage

### Generate Titles

**Endpoint:** `POST /generate`

**Request:**
```json
{
  "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
  "idea": "How modern warfare tactics evolved from historical battles"
}
```

**Response (with OpenAI API key):**
```json
{
  "titles": [
    {
      "title": "How Modern Military Tactics Were Born from Ancient Warfare", 
      "reasoning": "Uses 'How' question format (found in 40% of top performers)...",
      "confidence": 0.85
    }
  ],
  "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
  "idea": "How modern warfare tactics evolved from historical battles"
}
```

**Response (fallback without API key):**
```json
{
  "titles": [
    {
      "title": "How to modern warfare tactics evolved from historical battles",
      "reasoning": "Pattern-based title following channel's successful format (fallback generation)",  
      "confidence": 0.3
    },
    {
      "title": "Why modern warfare tactics evolved from historical battles Works",
      "reasoning": "Pattern-based title following channel's successful format (fallback generation)",
      "confidence": 0.3  
    }
  ]
}
```

## 🔧 Core Features

- **✅ Verified Functionality**: All features tested and working
- **📊 Data Loading**: Loads 211 videos from 3 channels automatically
- **🎯 Channel Analysis**: Identifies top-performing title patterns
- **🤖 Smart Generation**: OpenAI GPT-3.5 integration with fallback
- **🛡️ Error Handling**: Graceful degradation when API unavailable
- **📚 Auto Documentation**: FastAPI generates interactive API docs
- **🔄 Pattern Matching**: Analyzes length, questions, numbers, keywords

## � Testing the API

### Using curl:
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
       "idea": "The strategy behind famous military victories"
     }'
```

### Using Python:
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "channel_id": "UC510QYlOlKNyhy_zdQxnGYw", 
    "idea": "The strategy behind famous military victories"
})

print(response.json())
```

## 🎬 Available Test Channels

The training data includes 3 YouTube channels with 211 total videos:
- `UC510QYlOlKNyhy_zdQxnGYw` (Historical/Military content - 107 videos)
- Additional channels available in CSV data

## 🧪 Implementation Details

### Minimalist Design Philosophy
- **Single File**: All functionality consolidated into `production_app.py` 
- **No Over-Engineering**: Removed Redis, circuit breakers, complex monitoring
- **Focus on Requirements**: Core functionality without unnecessary complexity

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

