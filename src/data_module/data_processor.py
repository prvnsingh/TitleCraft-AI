import os
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


# Data Models
@dataclass
class VideoData:
    """Simple video data structure"""

    channel_id: str
    video_id: str
    title: str
    summary: str
    views_in_period: int


@dataclass
class ChannelAnalysis:
    """Channel performance analysis"""

    channel_id: str
    total_videos: int
    avg_views: float
    top_performers: List[VideoData]
    patterns: Dict[str, Any]


@dataclass
class GeneratedTitle:
    """Generated title with reasoning and metadata"""

    title: str
    reasoning: str
    confidence_score: float = 0.5
    model_used: str = "unknown"

    # Backward compatibility
    @property
    def confidence(self) -> float:
        """Backward compatibility for confidence field"""
        return self.confidence_score

    @confidence.setter
    def confidence(self, value: float):
        """Backward compatibility for confidence field"""
        self.confidence_score = value


# Data Loader Class
class DataLoader:
    """Simple CSV data loader and analyzer"""

    def __init__(
        self, csv_path: str = "electrify__applied_ai_engineer__training_data.csv"
    ):
        self.csv_path = csv_path
        self.data: Optional[pd.DataFrame] = None
        self.load_data()

    def load_data(self) -> None:
        """Load data from CSV file"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.data = pd.read_csv(self.csv_path)
        print(
            f"Loaded {len(self.data)} videos from {len(self.data['channel_id'].unique())} channels"
        )

    def get_channel_data(self, channel_id: str) -> List[VideoData]:
        """Get all videos for a specific channel"""
        if self.data is None:
            return []

        channel_videos = self.data[self.data["channel_id"] == channel_id]

        return [
            VideoData(
                channel_id=row["channel_id"],
                video_id=row["video_id"],
                title=row["title"],
                summary=row["summary"],
                views_in_period=row["views_in_period"],
            )
            for _, row in channel_videos.iterrows()
        ]

    def analyze_channel(self, channel_id: str) -> ChannelAnalysis:
        """Analyze channel performance patterns"""
        videos = self.get_channel_data(channel_id)

        if not videos:
            raise ValueError(f"No data found for channel: {channel_id}")

        # Basic statistics
        views = [v.views_in_period for v in videos]
        avg_views = sum(views) / len(views)

        # Get top performers (top 30%)
        sorted_videos = sorted(videos, key=lambda v: v.views_in_period, reverse=True)
        top_count = max(1, len(sorted_videos) // 3)
        top_performers = sorted_videos[:top_count]

        # Analyze title patterns from top performers
        patterns = self._analyze_title_patterns([v.title for v in top_performers])

        return ChannelAnalysis(
            channel_id=channel_id,
            total_videos=len(videos),
            avg_views=avg_views,
            top_performers=top_performers,
            patterns=patterns,
        )

    def _analyze_title_patterns(self, titles: List[str]) -> Dict[str, Any]:
        """Extract simple patterns from high-performing titles"""
        if not titles:
            return {}

        # Basic pattern analysis
        patterns = {
            "avg_length": sum(len(title.split()) for title in titles) / len(titles),
            "question_titles": sum(1 for title in titles if "?" in title) / len(titles),
            "numeric_titles": sum(
                1 for title in titles if any(char.isdigit() for char in title)
            )
            / len(titles),
            "exclamation_titles": sum(1 for title in titles if "!" in title)
            / len(titles),
            "common_words": self._get_common_words(titles),
            "sample_titles": titles[:3],  # Examples for pattern reference
        }

        return patterns

    def _get_common_words(self, titles: List[str]) -> List[str]:
        """Get most common words from titles (simple version)"""
        word_count = {}
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }

        for title in titles:
            words = title.lower().split()
            for word in words:
                # Clean word
                word = "".join(char for char in word if char.isalnum())
                if len(word) > 2 and word not in stop_words:
                    word_count[word] = word_count.get(word, 0) + 1

        # Return top 5 most common words
        return sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:5]
