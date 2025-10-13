# TitleCraft AI

Production-ready YouTube title generation system combining pattern analysis with multi-model LLM orchestration for high-performing content creation.

## 🚀 Quick Start

```bash
# Development setup
pip install -r requirements.txt
python -m src.api.production_app

# Production deployment  
docker-compose -f docker-compose.prod.yml up -d
```

## 🏗️ Architecture

**Production-grade system with:**
- Multi-LLM orchestration (OpenAI, Anthropic, Ollama, HuggingFace)
- Redis caching with circuit breakers
- Prometheus monitoring and structured logging
- FastAPI with rate limiting and validation
- Docker deployment with health checks

## 📁 Project Structure

```
src/
├── api/               # FastAPI production application
├── data/              # Data models and processing
├── processing/        # Pattern analysis and LLM orchestration
├── services/          # Business logic engines
├── infrastructure/    # Caching, monitoring, circuit breakers
└── config/            # Configuration management

tests/                 # Test suite
data/                  # Runtime data storage
```

## 🔧 Core Features

- **Channel profiling** with performance pattern extraction
- **Multi-model orchestration** with intelligent fallbacks
- **Adaptive prompting** based on channel characteristics
- **Pattern-guided generation** with explainable reasoning
- **Async caching** with Redis and circuit breakers
- **Comprehensive monitoring** with Prometheus metrics

## 🚦 API Endpoints

```bash
POST /api/v1/generate    # Generate titles for video content
GET  /api/v1/health      # System health and status
GET  /docs               # Interactive API documentation
```

## 🐳 Deployment

**Docker (Production)**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Local Development**  
```bash
python -m src.api.production_app
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Performance tests  
python -m pytest tests/test_performance.py -v
```

## � Security & Performance

- Input validation with Pydantic models
- Rate limiting and CORS protection
- Circuit breakers for API resilience
- Redis caching for <200ms response times
- Structured logging and Prometheus metrics

---

**Built for production scalability and reliability.**

