# TitleCraft AI - Data Module Documentation

## Overview

The TitleCraft AI Data Module provides comprehensive functionality for handling, analyzing, and profiling YouTube video data. It's designed to support the YouTube title generation system by extracting insights from successful video patterns.

## Module Structure

```
src/data/
├── __init__.py           # Module initialization and exports
├── models.py             # Data models and schemas
├── loader.py             # Data loading and preprocessing
├── validator.py          # Data quality validation
├── analyzer.py           # Statistical analysis and insights
├── profiler.py           # Channel profiling and pattern extraction
└── README.md             # This documentation
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

2. **Verify Installation**:
   ```bash
   python test_data_module.py
   ```

## Quick Start

### Basic Usage

```python
from src.data.loader import DataLoader
from src.data.analyzer import DataAnalyzer
from src.data.profiler import ChannelProfiler

# Load data
loader = DataLoader()
data = loader.load_from_csv("your_data.csv")

# Analyze data
analyzer = DataAnalyzer(data)
channel_stats = analyzer.analyze_channels()
performance_analysis = analyzer.analyze_performance()

# Create channel profiles
profiler = ChannelProfiler(data)
profiles = profiler.create_all_channel_profiles()

# Export results
profiler.export_profiles(profiles, "channel_profiles.json")
analyzer.export_analysis("analysis_results.json")
```

### Complete Example

Run the provided example script for a comprehensive demonstration:

```bash
python example_usage.py
```

## Core Components

### 1. Data Models (`models.py`)

Defines structured data classes for type-safe data handling:

- **VideoData**: Individual video information
- **ChannelStats**: Statistical summary of channel performance
- **ChannelProfile**: Comprehensive channel analysis profile
- **TitlePatterns**: Title format and pattern analysis
- **DataQualityReport**: Data validation results

### 2. Data Loader (`loader.py`)

**DataLoader Class** - Handles data ingestion and preprocessing:

```python
loader = DataLoader()

# Load from CSV with validation
data = loader.load_from_csv("data.csv")

# Get dataset summary
summary = loader.get_dataset_summary()

# Export processed data
loader.export_processed_data("output.csv")
```

**Key Features**:
- Automatic data type inference
- Missing value handling
- Data validation during load
- Export capabilities
- Comprehensive error handling

### 3. Data Validator (`validator.py`)

**DataValidator Class** - Ensures data quality and completeness:

```python
validator = DataValidator()
quality_report = validator.validate_dataset(data)

print(f"Quality Score: {quality_report.quality_score}")
print(f"Issues Found: {len(quality_report.issues)}")
```

**Validation Checks**:
- Missing value detection
- Data type validation
- Outlier identification
- Duplicate detection
- Content quality assessment
- Consistency verification

### 4. Data Analyzer (`analyzer.py`)

**DataAnalyzer Class** - Performs comprehensive statistical analysis:

```python
analyzer = DataAnalyzer(data)

# Channel analysis
channel_stats = analyzer.analyze_channels()

# Performance metrics
performance = analyzer.analyze_performance()

# Title pattern analysis
title_insights = analyzer.analyze_titles()

# Content theme analysis
content_analysis = analyzer.analyze_content()
```

**Analysis Features**:
- Channel performance categorization
- View distribution analysis
- Title pattern recognition
- Content theme extraction
- Correlation analysis
- Temporal trend analysis

### 5. Channel Profiler (`profiler.py`)

**ChannelProfiler Class** - Creates detailed channel profiles:

```python
profiler = ChannelProfiler(data)

# Single channel profile
profile = profiler.create_channel_profile("UC_CHANNEL_ID")

# All channels
all_profiles = profiler.create_all_channel_profiles()

# Export profiles
profiler.export_profiles(all_profiles, "profiles.json")
profiler.export_profile_summary(all_profiles, "summary.md")
```

**Profiling Features**:
- Channel type inference
- Success pattern identification
- Title format analysis
- High-performer extraction
- Success factor analysis
- Actionable recommendations

## Data Requirements

### Expected CSV Format

Your data file should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `channel_id` | string | YouTube channel identifier |
| `video_id` | string | YouTube video identifier |
| `title` | string | Video title |
| `summary` | string | Video description/summary |
| `views_in_period` | integer | View count for analysis period |

### Optional Columns

Additional columns that enhance analysis:
- `upload_date`: Video upload timestamp
- `duration`: Video length in seconds
- `category`: Video category
- `tags`: Video tags/keywords

## Output Formats

### 1. Channel Profiles (JSON)

```json
{
  "UC_CHANNEL_ID": {
    "channel_id": "UC_CHANNEL_ID",
    "channel_type": "space/science",
    "stats": {
      "total_videos": 45,
      "avg_views": 25000,
      "high_performer_threshold": 40000
    },
    "title_patterns": {
      "avg_length_words": 7.2,
      "question_ratio": 0.25,
      "numeric_ratio": 0.40,
      "emotional_hook_ratio": 0.15
    },
    "success_factors": {
      "optimal_title_length": {
        "recommendation": "Longer titles perform better (aim for ~8.5 words)"
      }
    }
  }
}
```

### 2. Analysis Results (JSON)

Comprehensive analysis results including:
- Dataset overview statistics
- Channel performance metrics
- Title pattern insights
- Content theme analysis
- Success correlations

### 3. Profile Summary (Markdown)

Human-readable summary with:
- Channel overviews
- Key performance metrics
- Top performing videos
- Success recommendations

## Advanced Usage

### Custom Analysis

```python
# Analyze specific channel
analyzer = DataAnalyzer(data)
channel_data = data[data['channel_id'] == 'YOUR_CHANNEL_ID']

# Custom performance thresholds
high_performers = analyzer.identify_high_performers(
    threshold_percentile=90
)

# Custom pattern analysis
title_patterns = analyzer.extract_title_patterns(
    min_frequency=5,
    pattern_types=['questions', 'superlatives', 'numbers']
)
```

### Batch Processing

```python
# Process multiple data files
data_files = ['data1.csv', 'data2.csv', 'data3.csv']
all_profiles = {}

for file in data_files:
    loader = DataLoader()
    data = loader.load_from_csv(file)
    
    profiler = ChannelProfiler(data)
    profiles = profiler.create_all_channel_profiles()
    all_profiles.update(profiles)

# Export combined results
profiler.export_profiles(all_profiles, 'combined_profiles.json')
```

## Testing

### Module Testing

Run the comprehensive test suite:

```bash
python test_data_module.py
```

### Sample Data

If you don't have data yet, the test script creates sample data:

```csv
channel_id,video_id,title,summary,views_in_period
UCTestChannel1,video1,Amazing Space Discovery: 10 Mind-Blowing Facts,Discover incredible facts about our universe,15000
UCTestChannel2,video2,Ultimate LEGO Tank Build - Military Series,Building an incredible LEGO military tank,8900
```

## Integration with TitleCraft AI

This data module is designed to integrate with the larger TitleCraft AI system:

1. **Data Preprocessing**: Clean and validate video data
2. **Pattern Extraction**: Identify successful title patterns
3. **Profile Creation**: Build channel-specific profiles
4. **LLM Integration**: Provide insights to title generation models
5. **API Integration**: Serve analysis results via FastAPI endpoints

## Performance Optimization

### Large Datasets

For datasets with millions of videos:

1. **Chunked Processing**:
   ```python
   loader = DataLoader()
   for chunk in loader.load_in_chunks("large_file.csv", chunk_size=10000):
       # Process chunk
       analyzer = DataAnalyzer(chunk)
       results = analyzer.analyze_performance()
   ```

2. **Memory Management**:
   - Use data filtering before analysis
   - Process channels individually
   - Clear intermediate results

3. **Parallel Processing**:
   ```python
   from multiprocessing import Pool
   
   def analyze_channel(channel_id):
       # Channel-specific analysis
       pass
   
   with Pool() as pool:
       results = pool.map(analyze_channel, channel_ids)
   ```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

2. **Memory Issues**:
   - Reduce chunk size
   - Filter data before processing
   - Use data sampling for large datasets

3. **Data Format Issues**:
   - Verify CSV column names
   - Check data types
   - Run data validation first

4. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Logging

Enable detailed logging for debugging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Now run your analysis with detailed logs
```

## Contributing

To extend the data module:

1. Follow existing code patterns
2. Add comprehensive error handling
3. Include type hints
4. Write documentation
5. Add tests for new functionality

## Support

For issues or questions:
1. Check this documentation
2. Run the test suite
3. Review the example usage
4. Examine the source code comments

The data module is designed to be robust, extensible, and production-ready for the TitleCraft AI system.