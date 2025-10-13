"""
Data models and schemas for TitleCraft AI.
Defines the structure and validation for data objects.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd


@dataclass
class VideoData:
    """Represents a single video entry"""
    channel_id: str
    video_id: str
    title: str
    summary: str
    views_in_period: int
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.views_in_period < 0:
            raise ValueError("Views cannot be negative")
        if not self.title.strip():
            raise ValueError("Title cannot be empty")


@dataclass
class ChannelStats:
    """Statistical summary for a channel"""
    channel_id: str
    total_videos: int
    avg_views: float
    median_views: float
    min_views: int
    max_views: int
    total_views: int
    std_views: float
    high_performer_threshold: float
    
    
@dataclass
class TitlePatterns:
    """Title pattern analysis results"""
    avg_length_words: float
    std_length_words: float
    avg_length_chars: float
    question_ratio: float
    numeric_ratio: float
    superlative_ratio: float
    emotional_hook_ratio: float
    punctuation_patterns: Dict[str, float]
    common_words: List[tuple]  # (word, frequency)
    common_bigrams: List[tuple]  # (bigram, frequency)
    common_trigrams: List[tuple]  # (trigram, frequency)


@dataclass
class ChannelProfile:
    """Complete channel profile with all analysis"""
    channel_id: str
    channel_type: str  # Inferred content type
    stats: ChannelStats
    title_patterns: TitlePatterns
    high_performers: List[VideoData]
    success_factors: Dict[str, Any]
    created_at: datetime
    data_version_hash: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'channel_id': self.channel_id,
            'channel_type': self.channel_type,
            'stats': self.stats.__dict__,
            'title_patterns': self.title_patterns.__dict__,
            'high_performers': [hp.__dict__ for hp in self.high_performers],
            'success_factors': self.success_factors,
            'created_at': self.created_at.isoformat(),
            'data_version_hash': self.data_version_hash
        }


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    total_records: int
    valid_records: int
    invalid_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    data_types_valid: bool
    outliers: Dict[str, List[Any]]
    quality_score: float  # 0-1 score
    issues: List[str]
    recommendations: List[str]


@dataclass
class DatasetSummary:
    """Overall dataset summary"""
    total_videos: int
    total_channels: int
    date_range: tuple  # (start, end) if available
    channel_distribution: Dict[str, int]
    view_distribution: Dict[str, float]  # percentiles
    quality_report: DataQualityReport
    insights: List[str]