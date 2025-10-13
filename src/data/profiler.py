"""
Channel profiler module for TitleCraft AI.
Creates detailed profiles for each YouTube channel based on successful patterns.
"""

# Standard library imports
import json
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .models import ChannelProfile, ChannelStats, TitlePatterns, VideoData
from .analyzer import DataAnalyzer

logger = logging.getLogger(__name__)


class ChannelProfiler:
    """
    Creates comprehensive profiles for YouTube channels based on their video performance patterns.
    """
    
    WORD_PATTERN = r'\b\w+\b'  # Regex pattern for word extraction
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize profiler with video data.
        
        Args:
            data: DataFrame containing video data with required columns
        """
        self.data = data.copy()
        self.analyzer = DataAnalyzer(data)
        self._validate_data()
        
    def _validate_data(self):
        """Validate that data has required columns."""
        required_cols = ['channel_id', 'video_id', 'title', 'summary', 'views_in_period']
        missing_cols = set(required_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")
    
    def create_all_channel_profiles(self) -> Dict[str, ChannelProfile]:
        """
        Create profiles for all channels in the dataset.
        
        Returns:
            Dictionary mapping channel_id to ChannelProfile
        """
        logger.info("Creating profiles for all channels...")
        
        profiles = {}
        channels = self.data['channel_id'].unique()
        
        for channel_id in channels:
            logger.info(f"Creating profile for channel: {channel_id}")
            profile = self.create_channel_profile(channel_id)
            profiles[channel_id] = profile
            
        logger.info(f"Created profiles for {len(profiles)} channels")
        return profiles
    
    def create_channel_profile(self, channel_id: str) -> ChannelProfile:
        """
        Create comprehensive profile for a specific channel.
        
        Args:
            channel_id: YouTube channel ID
            
        Returns:
            ChannelProfile object with complete analysis
        """
        # Get channel data
        channel_data = self.data[self.data['channel_id'] == channel_id].copy()
        
        if channel_data.empty:
            raise ValueError(f"No data found for channel: {channel_id}")
        
        # Calculate basic statistics
        stats = self._calculate_channel_stats(channel_data)
        
        # Analyze title patterns
        title_patterns = self._analyze_title_patterns(channel_data)
        
        # Identify high performers
        high_performers = self._identify_high_performers(channel_data, stats.high_performer_threshold)
        
        # Analyze success factors
        success_factors = self._analyze_success_factors(channel_data, high_performers)
        
        # Infer channel type
        channel_type = self._infer_channel_type(channel_data)
        
        # Create profile
        profile = ChannelProfile(
            channel_id=channel_id,
            channel_type=channel_type,
            stats=stats,
            title_patterns=title_patterns,
            high_performers=high_performers,
            success_factors=success_factors,
            created_at=datetime.now(),
            data_version_hash=self._calculate_data_hash(channel_data)
        )
        
        return profile
    
    def _calculate_channel_stats(self, channel_data: pd.DataFrame) -> ChannelStats:
        """Calculate basic statistics for channel."""
        views = channel_data['views_in_period']
        
        return ChannelStats(
            channel_id=channel_data['channel_id'].iloc[0],
            total_videos=len(channel_data),
            avg_views=float(views.mean()),
            median_views=float(views.median()),
            min_views=int(views.min()),
            max_views=int(views.max()),
            total_views=int(views.sum()),
            std_views=float(views.std()),
            high_performer_threshold=float(views.quantile(0.8))
        )
    
    def _analyze_title_patterns(self, channel_data: pd.DataFrame) -> TitlePatterns:
        """Analyze title patterns and characteristics."""
        titles = channel_data['title'].tolist()
        
        # Length analysis
        word_lengths = [len(title.split()) for title in titles]
        char_lengths = [len(title) for title in titles]
        
        # Pattern detection
        question_ratio = self._calculate_question_ratio(titles)
        numeric_ratio = self._calculate_numeric_ratio(titles)
        superlative_ratio = self._calculate_superlative_ratio(titles)
        emotional_ratio = self._calculate_emotional_ratio(titles)
        
        # Punctuation analysis
        punctuation_patterns = self._analyze_punctuation_patterns(titles)
        
        # N-gram analysis
        common_words = self._extract_common_words(titles, n=15)
        common_bigrams = self._extract_common_ngrams(titles, n=2, top_k=10)
        common_trigrams = self._extract_common_ngrams(titles, n=3, top_k=8)
        
        return TitlePatterns(
            avg_length_words=float(np.mean(word_lengths)),
            std_length_words=float(np.std(word_lengths)),
            avg_length_chars=float(np.mean(char_lengths)),
            question_ratio=question_ratio,
            numeric_ratio=numeric_ratio,
            superlative_ratio=superlative_ratio,
            emotional_hook_ratio=emotional_ratio,
            punctuation_patterns=punctuation_patterns,
            common_words=common_words,
            common_bigrams=common_bigrams,
            common_trigrams=common_trigrams
        )
    
    def _identify_high_performers(self, channel_data: pd.DataFrame, threshold: float) -> List[VideoData]:
        """Identify high-performing videos for the channel."""
        high_perf_data = channel_data[channel_data['views_in_period'] >= threshold]
        
        # Sort by views and take top performers
        high_perf_data = high_perf_data.sort_values('views_in_period', ascending=False)
        
        high_performers = []
        for _, row in high_perf_data.head(10).iterrows():  # Top 10 high performers
            video = VideoData(
                channel_id=row['channel_id'],
                video_id=row['video_id'],
                title=row['title'],
                summary=row['summary'],
                views_in_period=int(row['views_in_period'])
            )
            high_performers.append(video)
        
        return high_performers
    
    def _analyze_success_factors(self, channel_data: pd.DataFrame, 
                                high_performers: List[VideoData]) -> Dict[str, Any]:
        """Analyze what factors correlate with success for this channel."""
        
        if not high_performers:
            return {'message': 'No high performers identified'}
        
        high_perf_titles = [hp.title for hp in high_performers]
        all_titles = channel_data['title'].tolist()
        
        # Compare patterns between high performers and all videos
        factors = {}
        
        # Title length comparison
        high_perf_lengths = [len(title.split()) for title in high_perf_titles]
        all_lengths = [len(title.split()) for title in all_titles]
        
        factors['optimal_title_length'] = {
            'high_performers_avg': float(np.mean(high_perf_lengths)),
            'overall_avg': float(np.mean(all_lengths)),
            'recommendation': self._get_length_recommendation(high_perf_lengths, all_lengths)
        }
        
        # Pattern usage in high performers
        factors['successful_patterns'] = {
            'question_usage': self._calculate_question_ratio(high_perf_titles),
            'numeric_usage': self._calculate_numeric_ratio(high_perf_titles),
            'superlative_usage': self._calculate_superlative_ratio(high_perf_titles),
            'emotional_usage': self._calculate_emotional_ratio(high_perf_titles)
        }
        
        # Most successful words/phrases
        high_perf_words = self._extract_common_words(high_perf_titles, n=10)
        factors['power_words'] = high_perf_words
        
        # Successful title templates
        factors['successful_templates'] = self._extract_title_templates(high_perf_titles)
        
        return factors
    
    def _infer_channel_type(self, channel_data: pd.DataFrame) -> str:
        """Infer the type of content from titles and summaries."""
        all_text = ' '.join(channel_data['title'].tolist() + channel_data['summary'].fillna('').tolist())
        all_text = all_text.lower()
        
        # Define content type keywords
        content_indicators = {
            'military/war': ['war', 'military', 'tank', 'weapon', 'battle', 'soldier', 'army', 'combat', 'gun'],
            'space/science': ['space', 'solar', 'planet', 'nasa', 'universe', 'star', 'galaxy', 'science', 'discovery'],
            'gaming/toys': ['lego', 'game', 'toy', 'build', 'play', 'set', 'minifigure', 'brick'],
            'education': ['how', 'learn', 'explain', 'tutorial', 'guide', 'teach', 'education'],
            'entertainment': ['funny', 'entertainment', 'show', 'comedy', 'fun', 'viral'],
            'technology': ['tech', 'computer', 'software', 'digital', 'innovation', 'device']
        }
        
        # Count keyword matches
        type_scores = {}
        for content_type, keywords in content_indicators.items():
            score = sum(all_text.count(keyword) for keyword in keywords)
            type_scores[content_type] = score
        
        # Return most likely content type
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    # Pattern calculation methods
    def _calculate_question_ratio(self, titles: List[str]) -> float:
        """Calculate ratio of question-format titles."""
        question_count = sum(1 for title in titles if self._is_question(title))
        return question_count / len(titles) if titles else 0.0
    
    def _calculate_numeric_ratio(self, titles: List[str]) -> float:
        """Calculate ratio of titles containing numbers."""
        numeric_count = sum(1 for title in titles if re.search(r'\d', title))
        return numeric_count / len(titles) if titles else 0.0
    
    def _calculate_superlative_ratio(self, titles: List[str]) -> float:
        """Calculate ratio of titles containing superlatives."""
        superlatives = ['best', 'worst', 'most', 'least', 'biggest', 'smallest', 'fastest', 'slowest',
                       'greatest', 'ultimate', 'top', 'bottom', 'first', 'last', 'perfect', 'incredible']
        
        superlative_count = sum(1 for title in titles 
                               if any(word in title.lower() for word in superlatives))
        return superlative_count / len(titles) if titles else 0.0
    
    def _calculate_emotional_ratio(self, titles: List[str]) -> float:
        """Calculate ratio of titles containing emotional trigger words."""
        emotional_words = ['shocking', 'amazing', 'incredible', 'unbelievable', 'secret', 'hidden',
                          'revealed', 'exposed', 'never', 'always', 'everyone', 'nobody', 'insane',
                          'crazy', 'mind-blowing', 'epic', 'legendary', 'bizarre', 'mysterious']
        
        emotional_count = sum(1 for title in titles 
                             if any(word in title.lower() for word in emotional_words))
        return emotional_count / len(titles) if titles else 0.0
    
    def _analyze_punctuation_patterns(self, titles: List[str]) -> Dict[str, float]:
        """Analyze punctuation usage patterns."""
        total_titles = len(titles)
        if total_titles == 0:
            return {}
        
        punctuation_counts = {
            'exclamation': sum(title.count('!') for title in titles) / total_titles,
            'question': sum(title.count('?') for title in titles) / total_titles,
            'colon': sum(title.count(':') for title in titles) / total_titles,
            'dash': sum(title.count('-') for title in titles) / total_titles,
            'pipe': sum(title.count('|') for title in titles) / total_titles,
            'parentheses': sum(title.count('(') for title in titles) / total_titles,
            'brackets': sum(title.count('[') for title in titles) / total_titles
        }
        
        return punctuation_counts
    
    def _extract_common_words(self, titles: List[str], n: int = 15) -> List[tuple]:
        """Extract most common words from titles."""
        all_words = []
        for title in titles:
            words = re.findall(self.WORD_PATTERN, title.lower())
            # Filter out common stop words
            filtered_words = [word for word in words 
                            if word not in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}]
            all_words.extend(filtered_words)
        
        word_freq = Counter(all_words)
        return word_freq.most_common(n)
    
    def _extract_common_ngrams(self, titles: List[str], n: int = 2, top_k: int = 10) -> List[tuple]:
        """Extract common n-grams from titles."""
        ngrams = []
        for title in titles:
            words = re.findall(self.WORD_PATTERN, title.lower())
            if len(words) >= n:
                title_ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
                ngrams.extend(title_ngrams)
        
        ngram_freq = Counter(ngrams)
        return [(' '.join(ngram), count) for ngram, count in ngram_freq.most_common(top_k)]
    
    def _extract_title_templates(self, titles: List[str]) -> List[str]:
        """Extract common title structure templates."""
        templates = []
        
        # Common patterns to look for
        patterns = [
            r'^\d+\s+\w+',  # "5 Things", "10 Ways", etc.
            r'^(Why|How|What|When|Where)\s+',  # Question starters
            r'\w+\s+(vs|versus)\s+\w+',  # Comparison format
            r'^(The|A)\s+\w+\s+\w+',  # "The Best", "A Complete", etc.
            r'\w+\s+You\s+(Never|Didn\'t|Should)',  # "Things You Never", etc.
            r'^\w+\s+(Inside|Behind|Beyond)',  # "Life Inside", "Behind the", etc.
        ]
        
        for pattern in patterns:
            matches = [title for title in titles if re.search(pattern, title, re.IGNORECASE)]
            if matches:
                # Get the most common version of this pattern
                pattern_examples = [re.search(pattern, title, re.IGNORECASE).group() 
                                  for title in matches[:3]]
                templates.extend(pattern_examples)
        
        return list(set(templates))  # Remove duplicates
    
    # Helper methods
    def _is_question(self, title: str) -> bool:
        """Check if title is formatted as a question."""
        return ('?' in title or 
                re.match(r'^(why|how|what|when|where|who|which|can|is|are|do|does|did|will|would|should)', 
                        title.lower()))
    
    def _get_length_recommendation(self, high_perf_lengths: List[int], all_lengths: List[int]) -> str:
        """Generate recommendation for optimal title length."""
        high_avg = np.mean(high_perf_lengths)
        all_avg = np.mean(all_lengths)
        
        if high_avg > all_avg + 1:
            return f"Longer titles perform better (aim for ~{high_avg:.1f} words)"
        elif high_avg < all_avg - 1:
            return f"Shorter titles perform better (aim for ~{high_avg:.1f} words)"
        else:
            return f"Current average length ({all_avg:.1f} words) is optimal"
    
    def _calculate_data_hash(self, channel_data: pd.DataFrame) -> str:
        """Calculate hash for data versioning."""
        import hashlib
        data_string = channel_data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()[:16]
    
    # Export methods
    def export_profiles(self, profiles: Dict[str, ChannelProfile], output_path: str):
        """Export channel profiles to JSON file."""
        export_data = {}
        for channel_id, profile in profiles.items():
            export_data[channel_id] = profile.to_dict()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(profiles)} channel profiles to {output_path}")
    
    def export_profile_summary(self, profiles: Dict[str, ChannelProfile], output_path: str):
        """Export a human-readable summary of all profiles."""
        summary_lines = ["# Channel Profile Summary\n"]
        
        for channel_id, profile in profiles.items():
            summary_lines.extend([
                f"## Channel: {channel_id}",
                f"**Type**: {profile.channel_type}",
                f"**Videos**: {profile.stats.total_videos}",
                f"**Avg Views**: {profile.stats.avg_views:,.0f}",
                f"**High Performer Threshold**: {profile.stats.high_performer_threshold:,.0f}",
                "",
                "**Title Patterns**:",
                f"- Average Length: {profile.title_patterns.avg_length_words:.1f} words",
                f"- Question Format: {profile.title_patterns.question_ratio:.0%}",
                f"- Contains Numbers: {profile.title_patterns.numeric_ratio:.0%}",
                f"- Superlatives: {profile.title_patterns.superlative_ratio:.0%}",
                f"- Emotional Words: {profile.title_patterns.emotional_hook_ratio:.0%}",
                "",
                "**Top Performing Videos**:",
            ])
            
            for i, video in enumerate(profile.high_performers[:5], 1):
                summary_lines.append(f"{i}. {video.title} ({video.views_in_period:,} views)")
            
            summary_lines.extend(["", "---", ""])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Exported profile summary to {output_path}")