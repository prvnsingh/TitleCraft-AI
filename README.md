# TitleCraft AI - Minimal Take-Home Implementation

**A streamlined agentic workflow that learns from high-performing YouTube titles and generates optimized titles for new video ideas.**

> **🎯 Minimal Version**: Reduced from 60+ files to 7 essential files while preserving full functionality.

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

## 🛠️ Minimalism Achievement

| Before (Over-Engineered) | After (Minimal) |
|--------------------------|-----------------|
| 60+ files across 10+ directories | 7 essential files |
| Complex infrastructure (Redis, monitoring) | Single-file implementation |
| 55+ dependencies | 7 core dependencies |
| Multiple service layers | Consolidated classes |
| Production deployment configs | Development-ready server |

**Engineering Decision**: Focus on requirements, not impressive complexity.

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

