# TitleCraft AI - Take-Home Task

A simple agentic workflow that learns from high-performing YouTube titles and generates optimized titles for new video ideas.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup & Run

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Set OpenAI API Key**
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac  
export OPENAI_API_KEY=your_api_key_here
```

3. **Run the application**
```bash
python -m src.api.production_app
```

The API will be available at: http://localhost:8000

4. **View API documentation**
- Interactive docs: http://localhost:8000/docs
- API specification: http://localhost:8000/redoc

## 🏗️ Architecture

**Simple 3-layer workflow:**
```
CSV Data → Pattern Analysis → LLM Generation → API Response
    ↓              ↓              ↓         ↓
Data Loader    Channel         OpenAI     FastAPI
              Analysis         API        Endpoint
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

**Response:**
```json
{
  "titles": [
    {
      "title": "How Modern Military Tactics Were Born from Ancient Warfare",
      "reasoning": "Uses 'How' question format (found in 40% of top performers) and includes numbered elements. Matches channel's successful pattern of historical military analysis with modern relevance."
    },
    {
      "title": "Military Evolution: From Ancient Battlefields to Modern Combat",
      "reasoning": "Follows the channel's successful 'Topic: Comparison' format and uses common high-performing words like 'Military' and 'Combat' found in top titles."
    }
  ],
  "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
  "idea": "How modern warfare tactics evolved from historical battles"
}
```

## 🔧 Core Features

- **Channel-Specific Learning**: Analyzes patterns unique to each channel
- **Data-Grounded Reasoning**: Each title explains which successful patterns it uses
- **Pattern Analysis**: Identifies successful title characteristics (length, questions, numbers, common words)
- **LLM Integration**: Uses OpenAI GPT-3.5 with contextual prompting
- **Fallback Generation**: Works even if LLM is unavailable

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

The training data includes 3 YouTube channels. Use any of these channel IDs:
- `UC510QYlOlKNyhy_zdQxnGYw` (Historical/Military content)

## 🧪 Key Implementation Details

### Pattern Analysis
- Identifies top 30% performing titles per channel
- Analyzes: title length, question usage, numbers, exclamations, common words
- Uses patterns to guide LLM generation

### Title Generation
1. **Context Building**: Creates prompt with channel's successful patterns
2. **LLM Generation**: Uses OpenAI GPT-3.5 with pattern context  
3. **Reasoning**: Each title explains which patterns it leverages
4. **Confidence Scoring**: Based on pattern alignment

---

**Built for the Electrify Applied AI Engineer take-home task - focused on core functionality and clear presentation for CTO review.**

