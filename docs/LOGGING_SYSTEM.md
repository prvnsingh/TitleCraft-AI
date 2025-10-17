# TitleCraft AI - Structured Logging System

## 📋 Overview

TitleCraft AI uses a comprehensive structured logging system built around three specialized JSONL log files. This system provides complete visibility into data analytics, LLM operations, and API performance, making it ideal for analysis, debugging, and presentations.

## 🎯 Key Features

### ✅ **Three-Tier Structured Logging**
- **Data Analytics**: Metrics, patterns, mathematical insights
- **LLM Operations**: Prompt optimization, model interactions, token usage  
- **API Requests**: Endpoint performance, request/response cycles

### ✅ **JSONL Format**
- Each log entry is a complete JSON object on its own line
- Easy parsing with standard tools (jq, pandas, etc.)
- Consistent structure across all log types

### ✅ **Context-Aware Tracking**
- Request IDs for end-to-end request tracking
- Channel IDs for data-specific operations
- Operation IDs for complex multi-step processes

### ✅ **Presentation-Ready Data**
- Clear metrics and performance data
- Decision-making process visibility
- Mathematical calculations and confidence scores
- Token usage and cost tracking

## 📁 Log Files Structure

The structured logging system creates three specialized JSONL files in the `logs/` directory:

```
logs/
├── data_analytics.jsonl     # Data analysis, metrics, patterns (50MB, 5 backups)
├── llm_operations.jsonl     # LLM interactions, prompts, optimization (30MB, 5 backups)  
├── api_requests.jsonl       # API endpoints, performance, requests (20MB, 5 backups)
└── performance_YYYYMMDD.jsonl  # Legacy performance tracking (if enabled)
```

## 🔍 What Gets Logged

### 📊 **Data Analytics Log (`data_analytics.jsonl`)**
- **Pattern Discovery**: Channel classification, high-performer identification
- **Performance Metrics**: Response times, calculation results, confidence scores  
- **Mathematical Insights**: Statistical analysis, pattern confidence calculations
- **Channel Analysis**: Video performance data, title success metrics
- **Data Validation**: Loading status, data quality checks

### 🤖 **LLM Operations Log (`llm_operations.jsonl`)**  
- **Prompt Construction**: System/user prompts, data injection, optimization strategies
- **Model Interactions**: Request/response cycles, token usage, cost estimation
- **Parameter Adaptation**: Temperature, max tokens, strategy selection reasoning
- **Response Processing**: Content parsing, quality evaluation, enhancement steps
- **Error Handling**: Model failures, retry attempts, fallback strategies

### 🌐 **API Requests Log (`api_requests.jsonl`)**
- **Request Lifecycle**: Start/end timestamps, HTTP methods, endpoints
- **Performance Tracking**: Response times, status codes, payload sizes
- **Error Monitoring**: Failed requests, error types, debugging context
- **Service Integration**: External API calls, authentication, rate limiting

## 🚀 Usage Examples

### Making API Requests with Logging

The structured logging system automatically captures all activity when you make API requests:

```python
# Standard API usage - logging happens automatically
from src.services.title_generator import TitleGenerator, TitleGenerationRequest

generator = TitleGenerator()
request = TitleGenerationRequest(
    channel_id="UC510QYlOlKNyhy_zdQxnGYw", 
    video_idea="How to master Python programming",
    n_titles=3
)
response = generator.generate_titles(request)
```

### Analyzing Logs for Presentations

```bash
# View recent data analytics logs
tail -f logs/data_analytics.jsonl | jq '.'

# Filter LLM operations by model
jq 'select(.model=="gpt-4")' logs/llm_operations.jsonl

# Track API performance
jq 'select(.event=="api_request_end") | {endpoint, status_code, response_time}' logs/api_requests.jsonl

# Find pattern analysis results
jq 'select(.event=="pattern_analysis")' logs/data_analytics.jsonl
```

### Sample Log Entry Structure

```json
{
  "timestamp": "2025-10-17T07:22:35.382512Z",
  "level": "INFO",
  "module": "title_generator", 
  "function": "generate_titles",
  "line": 89,
  "message": "Starting title generation process",
  "component": "title_generator",
  "action": "generate_start",
  "video_idea": "How to master Python programming",
  "n_titles": 3,
  "model_name": "DeepSeek-R1-Distill-Qwen-32B",
  "request_id": "uuid-string",
  "channel_id": "UC510QYlOlKNyhy_zdQxnGYw"
}
```

## 🎤 Presentation Benefits

### **Code Flow Visibility**
- **Before**: "The AI analyzes the data..."
- **After**: "As shown in the logs, the system first loads 211 videos from 3 channels, discovers 15 key patterns with 87% confidence, then selects the 'high_volume_educational' strategy..."

### **Decision Reasoning**
- **Before**: "The AI picks the best titles..."  
- **After**: "The context-aware selector chose strategy X because the channel shows Y pattern with Z confidence, leading to temperature adjustment from 0.7 to 0.4..."

### **Performance Transparency**
- **Before**: "The system is fast..."
- **After**: "Pattern discovery took 0.13s, LLM generation 2.1s, quality evaluation 0.05s, with total response time of 2.28s using 1,247 tokens..."

### **Data Analysis Process**
- **Before**: "We analyze the channel data..."
- **After**: "The system identified 67% question usage, 4.2 average words, and top keywords ['Python', 'Tutorial', 'Beginner'] with pattern weights showing questions (0.34) and keywords (0.28) as strongest predictors..."

## 🛠️ Implementation Details

### Using Structured Logger

```python
from src.services.structured_logger import structured_logger

class MyService:
    def __init__(self):
        self.logger = structured_logger
        
    def process_data(self, data):
        # Log data analytics events
        self.logger.log_data_analytics({
            "event": "data_processing_start",
            "data_size": len(data),
            "processing_type": "analysis"
        })
```

### Decorator Usage

```python
from src.services.structured_logger import log_data_operation

@log_data_operation("channel_analysis", "data_processor")
def analyze_channel_data(channel_id):
    # Automatically logs operation with timing and results
    return analysis_results

@log_llm_operation("title_generation", "gpt-4")  
def generate_titles(prompt):
    # Logs LLM operations with model info
    return generated_titles
```

### Manual Logging Examples

```python
# Data Analytics Logging
structured_logger.log_pattern_analysis(
    patterns={"keywords": ["python", "tutorial"], "avg_views": 15000},
    confidence=0.87,
    channel_data={"total_videos": 150, "category": "education"}
)

# LLM Operations Logging
structured_logger.log_llm_request(
    model="gpt-4",
    parameters={"temperature": 0.7, "max_tokens": 1500},
    token_estimate=1200
)

# API Request Logging
structured_logger.log_api_request_start(
    endpoint="/generate-titles",
    method="POST", 
    request_data={"video_idea": "Python tutorial", "n_titles": 5}
)
```

## 🔧 Configuration

### Log File Settings

The system automatically creates and manages three JSONL log files with the following rotation settings:

- **data_analytics.jsonl**: 50MB max size, 5 backup files
- **llm_operations.jsonl**: 30MB max size, 5 backup files  
- **api_requests.jsonl**: 20MB max size, 5 backup files

### Customization

The logging system can be customized by modifying `src/services/structured_logger.py`:

- Adjust log file sizes and backup counts
- Add new specialized loggers
- Create custom logging methods
- Modify JSONL entry structure
- Add new decorators for automatic logging

## 📊 Log Analysis Tips

### For Presentations

1. **Data Flow**: Show pattern discovery and analysis insights from `data_analytics.jsonl`
2. **LLM Decisions**: Highlight prompt strategies and model reasoning from `llm_operations.jsonl`
3. **Performance Metrics**: Extract response times and token usage across all logs
4. **End-to-End Tracking**: Follow request_id through all three log files

### For Debugging

1. **Cross-Log Analysis**: Correlate events across data/LLM/API logs using request_id
2. **Performance Bottlenecks**: Identify slow operations in data processing or LLM calls
3. **Error Patterns**: Analyze error events and their contexts
4. **Token Usage**: Monitor LLM costs and usage patterns over time

### Sample Analysis Commands

```bash
# Find all events for a specific request
grep '"request_id":"abc123"' logs/*.jsonl | jq '.'

# Get token usage summary
jq 'select(.tokens_used) | {model, tokens_used, cost}' logs/llm_operations.jsonl

# Performance analysis by event type  
jq 'select(.response_time) | {event, response_time}' logs/api_requests.jsonl | jq -s 'sort_by(.response_time)'
```

## 🎯 Conclusion

This comprehensive logging system transforms TitleCraft AI from a "black box" into a fully transparent, observable system. Every decision, every analysis step, and every performance metric is captured in structured, queryable logs that make presentations compelling and debugging straightforward.

The logs provide the narrative of how AI reasoning works in practice, making technical presentations more engaging and credible with concrete data and visible decision-making processes.