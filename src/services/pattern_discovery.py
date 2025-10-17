"""
Pattern Discovery Agent
Intelligently analyzes channel performance data and dynamically weights pattern importance
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np

from src.data_module.data_processor import VideoData
from .structured_logger import structured_logger, log_data_operation


@dataclass
class PatternWeights:
    """Weights for different pattern types based on their predictive power"""
    word_count_weight: float
    question_weight: float
    numeric_weight: float
    exclamation_weight: float
    capitalization_weight: float
    keyword_weight: float
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = (self.word_count_weight + self.question_weight + self.numeric_weight + 
                self.exclamation_weight + self.capitalization_weight + self.keyword_weight)
        
        # Check for zero, NaN, or negative total
        if total > 0 and not np.isnan(total):
            self.word_count_weight /= total
            self.question_weight /= total
            self.numeric_weight /= total
            self.exclamation_weight /= total
            self.capitalization_weight /= total
            self.keyword_weight /= total
        else:
            # Use equal weights if total is zero, NaN, or negative
            equal_weight = 1.0 / 6.0
            self.word_count_weight = equal_weight
            self.question_weight = equal_weight
            self.numeric_weight = equal_weight
            self.exclamation_weight = equal_weight
            self.capitalization_weight = equal_weight
            self.keyword_weight = equal_weight


@dataclass
class IntelligentPatterns:
    """Enhanced pattern analysis with predictive weights"""
    avg_word_count: float
    question_percentage: float
    numeric_percentage: float
    exclamation_percentage: float
    capitalization_score: float
    top_keywords: List[str]
    pattern_weights: PatternWeights
    channel_type: str  # "high_volume", "medium_volume", "low_volume"
    content_style: str  # "educational", "entertainment", "mixed"
    confidence_score: float


class PatternDiscoveryAgent:
    """
    Intelligent pattern discovery that adapts to channel characteristics
    """
    
    def __init__(self):
        self.logger = structured_logger
        self.performance_threshold_percentile = 70  # Top 30% are considered high-performing
        
        self.logger.log_data_analytics({
            "event": "pattern_discovery_agent_initialized",
            "performance_threshold": self.performance_threshold_percentile,
            "component": "pattern_discovery"
        })
    
    @log_data_operation("pattern_discovery", "pattern_discovery")
    def discover_patterns(self, videos: List[VideoData]) -> IntelligentPatterns:
        """
        Intelligently discover patterns with adaptive weighting
        """
        self.logger.info("Starting pattern discovery", extra={
            'extra_fields': {
                'component': 'pattern_discovery',
                'action': 'discovery_start',
                'total_videos': len(videos)
            }
        })
        
        if not videos:
            self.logger.warning("No videos provided, using default patterns", extra={
                'extra_fields': {
                    'component': 'pattern_discovery',
                    'action': 'no_videos_fallback'
                }
            })
            return self._get_default_patterns()
        
        # Classify channel type and content style
        channel_type = self._classify_channel_type(videos)
        content_style = self._classify_content_style(videos)
        
        self.logger.info("Channel classification completed", extra={
            'extra_fields': {
                'component': 'pattern_discovery',
                'action': 'channel_classification',
                'channel_type': channel_type,
                'content_style': content_style
            }
        })
        
        # Get high-performing titles
        high_performers = self._identify_high_performers(videos)
        
        self.logger.info("High performers identified", extra={
            'extra_fields': {
                'component': 'pattern_discovery',
                'action': 'high_performers_identified',
                'high_performers_count': len(high_performers),
                'high_performer_ratio': len(high_performers) / len(videos) if videos else 0
            }
        })
        
        # Extract base patterns
        patterns = self._extract_base_patterns(high_performers)
        
        # Calculate pattern weights based on performance correlation
        weights = self._calculate_pattern_weights(videos)
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(videos, high_performers)
        
        return IntelligentPatterns(
            avg_word_count=patterns['avg_word_count'],
            question_percentage=patterns['question_percentage'],
            numeric_percentage=patterns['numeric_percentage'],
            exclamation_percentage=patterns['exclamation_percentage'],
            capitalization_score=patterns['capitalization_score'],
            top_keywords=patterns['top_keywords'],
            pattern_weights=weights,
            channel_type=channel_type,
            content_style=content_style,
            confidence_score=confidence
        )
    
    def _classify_channel_type(self, videos: List[VideoData]) -> str:
        """Classify channel based on volume and performance characteristics"""
        total_videos = len(videos)
        avg_views = np.mean([v.views_in_period for v in videos])
        
        if total_videos >= 50 and avg_views > 10000:
            return "high_volume"
        elif total_videos >= 20 and avg_views > 1000:
            return "medium_volume"
        else:
            return "low_volume"
    
    def _classify_content_style(self, videos: List[VideoData]) -> str:
        """Classify content style based on title patterns"""
        educational_keywords = ['how', 'tutorial', 'guide', 'learn', 'tips', 'explained', 'review']
        entertainment_keywords = ['funny', 'amazing', 'incredible', 'shocking', 'unbelievable', 'epic']
        
        educational_score = 0
        entertainment_score = 0
        
        for video in videos:
            title_lower = video.title.lower()
            educational_score += sum(1 for keyword in educational_keywords if keyword in title_lower)
            entertainment_score += sum(1 for keyword in entertainment_keywords if keyword in title_lower)
        
        total_videos = len(videos)
        edu_ratio = educational_score / total_videos if total_videos > 0 else 0
        ent_ratio = entertainment_score / total_videos if total_videos > 0 else 0
        
        if edu_ratio > ent_ratio * 1.5:
            return "educational"
        elif ent_ratio > edu_ratio * 1.5:
            return "entertainment"
        else:
            return "mixed"
    
    def _identify_high_performers(self, videos: List[VideoData]) -> List[VideoData]:
        """Identify high-performing videos using adaptive thresholds"""
        if len(videos) < 3:
            return videos  # Use all videos if dataset is small
        
        views = [v.views_in_period for v in videos]
        threshold = np.percentile(views, self.performance_threshold_percentile)
        
        high_performers = [v for v in videos if v.views_in_period >= threshold]
        
        # Ensure we have at least 20% of videos as high performers
        min_performers = max(1, len(videos) // 5)
        if len(high_performers) < min_performers:
            sorted_videos = sorted(videos, key=lambda x: x.views_in_period, reverse=True)
            high_performers = sorted_videos[:min_performers]
        
        return high_performers
    
    def _extract_base_patterns(self, videos: List[VideoData]) -> Dict[str, Any]:
        """Extract basic pattern metrics"""
        if not videos:
            return self._get_default_pattern_dict()
        
        titles = [v.title for v in videos]
        
        # Word count analysis
        word_counts = [len(title.split()) for title in titles]
        avg_word_count = np.mean(word_counts)
        
        # Pattern percentages
        question_count = sum(1 for title in titles if '?' in title)
        numeric_count = sum(1 for title in titles if re.search(r'\d+', title))
        exclamation_count = sum(1 for title in titles if '!' in title)
        
        total = len(titles)
        question_percentage = question_count / total if total > 0 else 0
        numeric_percentage = numeric_count / total if total > 0 else 0
        exclamation_percentage = exclamation_count / total if total > 0 else 0
        
        # Capitalization analysis
        capitalization_scores = []
        for title in titles:
            words = title.split()
            if words:
                capitalized_words = sum(1 for word in words if word[0].isupper())
                capitalization_scores.append(capitalized_words / len(words))
        
        capitalization_score = np.mean(capitalization_scores) if capitalization_scores else 0
        
        # Keyword extraction
        top_keywords = self._extract_keywords(titles)
        
        return {
            'avg_word_count': avg_word_count,
            'question_percentage': question_percentage,
            'numeric_percentage': numeric_percentage,
            'exclamation_percentage': exclamation_percentage,
            'capitalization_score': capitalization_score,
            'top_keywords': top_keywords
        }
    
    def _calculate_pattern_weights(self, all_videos: List[VideoData]) -> PatternWeights:
        """Calculate pattern weights based on performance correlation"""
        if len(all_videos) < 5:
            return self._get_default_weights()
        
        # Calculate correlation between patterns and performance
        views = [v.views_in_period for v in all_videos]
        
        # Word count correlation
        word_counts = [len(v.title.split()) for v in all_videos]
        try:
            word_count_corr = abs(np.corrcoef(word_counts, views)[0, 1]) if len(set(word_counts)) > 1 else 0.1
            word_count_corr = 0.1 if np.isnan(word_count_corr) else word_count_corr
        except (ValueError, IndexError):
            word_count_corr = 0.1
        
        # Question mark correlation
        question_flags = [1 if '?' in v.title else 0 for v in all_videos]
        try:
            question_corr = abs(np.corrcoef(question_flags, views)[0, 1]) if len(set(question_flags)) > 1 else 0.1
            question_corr = 0.1 if np.isnan(question_corr) else question_corr
        except (ValueError, IndexError):
            question_corr = 0.1
        
        # Numeric correlation
        numeric_flags = [1 if re.search(r'\d+', v.title) else 0 for v in all_videos]
        try:
            numeric_corr = abs(np.corrcoef(numeric_flags, views)[0, 1]) if len(set(numeric_flags)) > 1 else 0.1
            numeric_corr = 0.1 if np.isnan(numeric_corr) else numeric_corr
        except (ValueError, IndexError):
            numeric_corr = 0.1
        
        # Exclamation correlation
        exclamation_flags = [1 if '!' in v.title else 0 for v in all_videos]
        try:
            exclamation_corr = abs(np.corrcoef(exclamation_flags, views)[0, 1]) if len(set(exclamation_flags)) > 1 else 0.1
            exclamation_corr = 0.1 if np.isnan(exclamation_corr) else exclamation_corr
        except (ValueError, IndexError):
            exclamation_corr = 0.1
        
        # Capitalization correlation
        cap_scores = []
        for v in all_videos:
            words = v.title.split()
            if words:
                cap_scores.append(sum(1 for word in words if word[0].isupper()) / len(words))
            else:
                cap_scores.append(0)
        try:
            cap_corr = abs(np.corrcoef(cap_scores, views)[0, 1]) if len(set(cap_scores)) > 1 else 0.1
            cap_corr = 0.1 if np.isnan(cap_corr) else cap_corr
        except (ValueError, IndexError):
            cap_corr = 0.1
        
        # Default keyword weight
        keyword_weight = 0.2
        
        weights = PatternWeights(
            word_count_weight=word_count_corr,
            question_weight=question_corr,
            numeric_weight=numeric_corr,
            exclamation_weight=exclamation_corr,
            capitalization_weight=cap_corr,
            keyword_weight=keyword_weight
        )
        
        weights.normalize()
        return weights
    
    def _extract_keywords(self, titles: List[str]) -> List[str]:
        """Extract top keywords from titles"""
        # Common stop words to exclude
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                     'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                     'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        # Extract words and count frequency
        all_words = []
        for title in titles:
            words = re.findall(r'\b\w+\b', title.lower())
            words = [w for w in words if len(w) > 2 and w not in stop_words]
            all_words.extend(words)
        
        # Get top 10 most common keywords
        word_counts = Counter(all_words)
        top_keywords = [word for word, count in word_counts.most_common(10)]
        
        return top_keywords
    
    def _calculate_confidence(self, all_videos: List[VideoData], 
                           high_performers: List[VideoData]) -> float:
        """Calculate confidence score based on data quality"""
        total_videos = len(all_videos)
        high_performer_count = len(high_performers)
        
        # Base confidence on data volume
        volume_confidence = min(1.0, total_videos / 30)  # Full confidence at 30+ videos
        
        # Confidence based on performance distribution
        if total_videos > 1:
            views = [v.views_in_period for v in all_videos]
            view_std = np.std(views)
            view_mean = np.mean(views)
            cv = view_std / view_mean if view_mean > 0 else 1.0 # coefficient of variance
            distribution_confidence = max(0.3, min(1.0, 2.0 - cv))  # Lower CV = higher confidence
        else:
            distribution_confidence = 0.3
        
        # Confidence based on high performer ratio
        ratio_confidence = min(1.0, high_performer_count / max(1, total_videos * 0.2))
        
        # Combined confidence
        overall_confidence = (volume_confidence * 0.4 + 
                            distribution_confidence * 0.4 + 
                            ratio_confidence * 0.2)
        
        return max(0.1, min(1.0, overall_confidence))
    
    def _get_default_patterns(self) -> IntelligentPatterns:
        """Return default patterns for empty datasets"""
        return IntelligentPatterns(
            avg_word_count=8.0,
            question_percentage=0.2,
            numeric_percentage=0.3,
            exclamation_percentage=0.1,
            capitalization_score=0.6,
            top_keywords=['how', 'best', 'tips', 'guide', 'amazing'],
            pattern_weights=self._get_default_weights(),
            channel_type="low_volume",
            content_style="mixed",
            confidence_score=0.1
        )
    
    def _get_default_pattern_dict(self) -> Dict[str, Any]:
        """Return default pattern dictionary"""
        return {
            'avg_word_count': 8.0,
            'question_percentage': 0.2,
            'numeric_percentage': 0.3,
            'exclamation_percentage': 0.1,
            'capitalization_score': 0.6,
            'top_keywords': ['how', 'best', 'tips', 'guide', 'amazing']
        }
    
    def _get_default_weights(self) -> PatternWeights:
        """Return default balanced weights"""
        weights = PatternWeights(
            word_count_weight=0.2,
            question_weight=0.15,
            numeric_weight=0.2,
            exclamation_weight=0.1,
            capitalization_weight=0.15,
            keyword_weight=0.2
        )
        weights.normalize()
        return weights