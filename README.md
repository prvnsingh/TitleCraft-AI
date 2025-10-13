# TitleCraft AI

A hybrid AI system that combines pattern analysis with LLM generation to create high-performing YouTube video titles. This project implements a data-grounded approach to title generation by learning from successful patterns in existing content.

## Overview

TitleCraft AI analyzes high-performing video titles across different channels to extract lightweight, interpretable patterns. These patterns are then used to condition an LLM generator to create compelling new titles with data-backed reasoning.

## Features

- **Pattern Analysis**: Extracts successful title patterns from high-performing content
- **LLM Integration**: Uses extracted patterns to guide intelligent title generation
- **Channel-Specific**: Learns and generates titles tailored to specific channel characteristics
- **Data-Grounded Reasoning**: Provides explanations for why each generated title should perform well
- **FastAPI Endpoint**: Ready-to-deploy API for title generation

## Project Structure

```
├── electrify__applied_ai_engineer__training_data.csv  # Training dataset
├── implementation_plan_hybrid_pattern_llm.md         # Detailed implementation plan
├── Applied AI Engineer _ Take-Home Task.pdf          # Original task specification
└── README.md                                         # This file
```

## Architecture

The system uses a hybrid approach:

1. **Offline Data Profiler**: Analyzes existing title performance to extract patterns
2. **Pattern Extraction**: Identifies successful linguistic and structural patterns per channel  
3. **LLM Generator**: Creates new titles using extracted patterns as conditioning
4. **Reasoning Engine**: Provides data-backed explanations for generated titles

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Required ML/NLP libraries (see implementation plan for details)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd TitleCraft-AI

# Install dependencies (to be added)
pip install -r requirements.txt

# Run the API (to be implemented)
uvicorn main:app --reload
```

## Usage

The API will provide endpoints to:
- Generate 3-5 candidate titles for new video ideas
- Get pattern analysis for specific channels
- Receive data-grounded reasoning for title recommendations

