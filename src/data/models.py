"""
Data models and schemas for TitleCraft AI.
Defines the structure and validation for data objects.
"""

# Standard library imports
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any


@dataclass
class VideoData:
    """
    Represents a single video entry with validation.
    
    Attributes:
        channel_id: YouTube channel ID (format: UC...)
        video_id: YouTube video ID  
        title: Video title (1-100 characters)
        summary: Video description/summary
        views_in_period: Number of views in the analysis period (≥0)
    """
    channel_id: str
    video_id: str
    title: str
    summary: str
    views_in_period: int
    
    def __post_init__(self):
        """Validate data after initialization"""
        # Validate views
        if self.views_in_period < 0:
            raise ValueError("Views cannot be negative")
            
        # Validate title
        if not self.title or not self.title.strip():
            raise ValueError("Title cannot be empty")
        if len(self.title) > 100:
            raise ValueError("Title cannot exceed 100 characters")
            
        # Validate channel_id format (basic check)
        if not self.channel_id or len(self.channel_id) < 5:
            raise ValueError("Invalid channel_id format")
            
        # Validate video_id format (basic check)  
        if not self.video_id or len(self.video_id) < 5:
            raise ValueError("Invalid video_id format")
            
        # Clean up whitespace
        self.title = self.title.strip()
        self.summary = self.summary.strip() if self.summary else ""


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
    """
    Complete channel profile with all analysis and validation.
    
    Represents a comprehensive analysis of a YouTube channel including
    statistical performance data, title patterns, and success factors.
    """
    channel_id: str
    channel_type: str  # Inferred content type
    stats: ChannelStats
    title_patterns: TitlePatterns
    high_performers: List[VideoData]
    success_factors: Dict[str, Any]
    created_at: datetime
    data_version_hash: str
    
    def __post_init__(self):
        """Validate channel profile data after initialization"""
        if not self.channel_id:
            raise ValueError("Channel ID is required")
        if not self.channel_type:
            raise ValueError("Channel type is required")
        if not self.data_version_hash:
            raise ValueError("Data version hash is required")
        if not self.high_performers:
            self.high_performers = []
        if not self.success_factors:
            self.success_factors = {}
    
    def to_dict(self) -> Dict[str, Any]:
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
    
    @property
    def performance_category(self) -> str:
        """Get performance category based on average views"""
        avg_views = self.stats.avg_views
        if avg_views > 50000:
            return "high"
        elif avg_views > 10000:
            return "medium"
        else:
            return "low"


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