# Data Module for TitleCraft AI

This module handles all data-related operations including:
- Raw data ingestion and validation
- Data quality assessment
- Channel-specific analysis
- Pattern profiling
- Statistical analysis
- Export functionality

## Module Structure

```
src/data/
├── __init__.py
├── loader.py          # Data loading and validation
├── analyzer.py        # Statistical analysis and insights
├── profiler.py        # Channel pattern profiling
├── validator.py       # Data quality validation
├── exporter.py        # Data export utilities
└── models.py          # Data models and schemas
```

## Usage

```python
from src.data.loader import DataLoader
from src.data.analyzer import DataAnalyzer
from src.data.profiler import ChannelProfiler

# Load and analyze data
loader = DataLoader('electrify__applied_ai_engineer__training_data.csv')
data = loader.load_and_validate()

# Perform analysis
analyzer = DataAnalyzer(data)
insights = analyzer.generate_comprehensive_analysis()

# Create channel profiles
profiler = ChannelProfiler(data)
profiles = profiler.create_all_channel_profiles()
```