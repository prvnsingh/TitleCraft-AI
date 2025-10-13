"""
Data analyzer module for TitleCraft AI.
Performs statistical analysis and generates insights from YouTube video data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from collections import Counter
import logging

from .models import ChannelStats, DatasetSummary

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Analyzes YouTube video data to extract statistical insights and patterns.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize analyzer with video data.
        
        Args:
            data: DataFrame containing video data
        """
        self.data = data.copy()
        self._validate_data()
        
    def _validate_data(self):
        """Validate that data has required columns."""
        required_cols = ['channel_id', 'video_id', 'title', 'summary', 'views_in_period']
        missing_cols = set(required_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis of the dataset.
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Generating comprehensive data analysis...")
        
        analysis = {
            'dataset_overview': self.get_dataset_overview(),
            'channel_analysis': self.analyze_channels(),
            'performance_analysis': self.analyze_performance(),
            'title_analysis': self.analyze_titles(),
            'content_analysis': self.analyze_content(),
            'correlation_analysis': self.analyze_correlations(),
            'insights': self.generate_insights()
        }
        
        logger.info("Analysis complete")
        return analysis
    
    def get_dataset_overview(self) -> Dict[str, Any]:
        """Get high-level dataset statistics."""
        return {
            'total_videos': len(self.data),
            'unique_channels': self.data['channel_id'].nunique(),
            'unique_videos': self.data['video_id'].nunique(),
            'date_range': None,  # No date column in current data
            'data_completeness': {
                col: (self.data[col].notna().sum() / len(self.data)) * 100 
                for col in self.data.columns
            }
        }
    
    def analyze_channels(self) -> Dict[str, Any]:
        """Analyze channel-specific patterns and statistics."""
        channel_stats = {}
        
        for channel_id in self.data['channel_id'].unique():
            channel_data = self.data[self.data['channel_id'] == channel_id]
            channel_stats[channel_id] = self._calculate_channel_stats(channel_data)
        
        # Overall channel analysis
        channel_sizes = self.data['channel_id'].value_counts()
        
        return {
            'individual_channels': channel_stats,
            'channel_distribution': {
                'largest_channel': {
                    'id': channel_sizes.index[0],
                    'video_count': int(channel_sizes.iloc[0])
                },
                'smallest_channel': {
                    'id': channel_sizes.index[-1], 
                    'video_count': int(channel_sizes.iloc[-1])
                },
                'size_distribution': {
                    'mean': float(channel_sizes.mean()),
                    'std': float(channel_sizes.std()),
                    'sizes': channel_sizes.to_dict()
                }
            }
        }
    
    def _calculate_channel_stats(self, channel_data: pd.DataFrame) -> ChannelStats:
        """Calculate detailed statistics for a single channel."""
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
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze view performance patterns."""
        views = self.data['views_in_period']
        
        # Performance tiers
        high_perf = views.quantile(0.9)
        medium_perf = views.quantile(0.7)
        low_perf = views.quantile(0.3)
        
        # Performance by channel
        channel_performance = {}
        for channel_id in self.data['channel_id'].unique():
            channel_data = self.data[self.data['channel_id'] == channel_id]
            channel_views = channel_data['views_in_period']
            
            channel_performance[channel_id] = {
                'avg_views': float(channel_views.mean()),
                'high_performers_count': int((channel_views >= high_perf).sum()),
                'low_performers_count': int((channel_views <= low_perf).sum()),
                'consistency': float(1 / (1 + channel_views.std() / channel_views.mean())),  # Inverse CV
                'top_video': {
                    'title': channel_data.loc[channel_views.idxmax(), 'title'],
                    'views': int(channel_views.max())
                }
            }
        
        return {
            'overall_performance': {
                'mean_views': float(views.mean()),
                'median_views': float(views.median()),
                'std_views': float(views.std()),
                'min_views': int(views.min()),
                'max_views': int(views.max()),
                'percentiles': {
                    'p10': float(views.quantile(0.1)),
                    'p25': float(views.quantile(0.25)),
                    'p75': float(views.quantile(0.75)),
                    'p90': float(views.quantile(0.9)),
                    'p95': float(views.quantile(0.95)),
                    'p99': float(views.quantile(0.99))
                }
            },
            'performance_tiers': {
                'high_threshold': float(high_perf),
                'medium_threshold': float(medium_perf),
                'low_threshold': float(low_perf),
                'high_count': int((views >= high_perf).sum()),
                'medium_count': int(((views >= medium_perf) & (views < high_perf)).sum()),
                'low_count': int((views <= low_perf).sum())
            },
            'channel_performance': channel_performance
        }
    
    def analyze_titles(self) -> Dict[str, Any]:
        """Analyze title patterns and characteristics."""
        titles = self.data['title'].tolist()
        
        # Basic title metrics
        title_lengths_words = [len(title.split()) for title in titles]
        title_lengths_chars = [len(title) for title in titles]
        
        # Pattern analysis
        question_count = sum(1 for title in titles if self._is_question(title))
        numeric_count = sum(1 for title in titles if self._contains_numbers(title))
        superlative_count = sum(1 for title in titles if self._contains_superlatives(title))
        emotional_count = sum(1 for title in titles if self._contains_emotional_words(title))
        
        # Word frequency analysis
        all_words = []
        for title in titles:
            words = re.findall(r'\b\w+\b', title.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        common_words = word_freq.most_common(20)
        
        # Bigram analysis
        bigrams = []
        for title in titles:
            words = re.findall(r'\b\w+\b', title.lower())
            bigrams.extend(zip(words[:-1], words[1:]))
        
        bigram_freq = Counter(bigrams)
        common_bigrams = [(' '.join(bigram), count) for bigram, count in bigram_freq.most_common(10)]
        
        return {
            'length_analysis': {
                'avg_words': np.mean(title_lengths_words),
                'std_words': np.std(title_lengths_words),
                'min_words': min(title_lengths_words),
                'max_words': max(title_lengths_words),
                'avg_chars': np.mean(title_lengths_chars),
                'std_chars': np.std(title_lengths_chars)
            },
            'pattern_analysis': {
                'question_ratio': question_count / len(titles),
                'numeric_ratio': numeric_count / len(titles), 
                'superlative_ratio': superlative_count / len(titles),
                'emotional_ratio': emotional_count / len(titles)
            },
            'word_frequency': {
                'most_common_words': common_words,
                'most_common_bigrams': common_bigrams,
                'unique_words': len(word_freq),
                'total_words': len(all_words)
            },
            'punctuation_analysis': self._analyze_punctuation(titles)
        }
    
    def analyze_content(self) -> Dict[str, Any]:
        """Analyze video content patterns from summaries."""
        summaries = self.data['summary'].fillna('').tolist()
        
        # Summary length analysis
        summary_lengths = [len(summary.split()) for summary in summaries if summary]
        
        # Content keywords from summaries
        summary_words = []
        for summary in summaries:
            if summary:
                words = re.findall(r'\b\w+\b', summary.lower())
                summary_words.extend(words)
        
        summary_word_freq = Counter(summary_words)
        
        return {
            'summary_analysis': {
                'avg_length': np.mean(summary_lengths) if summary_lengths else 0,
                'std_length': np.std(summary_lengths) if summary_lengths else 0,
                'empty_summaries': sum(1 for s in summaries if not s.strip()),
                'common_content_words': summary_word_freq.most_common(15)
            }
        }
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between title features and performance."""
        
        # Create feature matrix
        features = []
        views = []
        
        for _, row in self.data.iterrows():
            title = row['title']
            title_features = {
                'word_count': len(title.split()),
                'char_count': len(title),
                'is_question': int(self._is_question(title)),
                'has_numbers': int(self._contains_numbers(title)),
                'has_superlatives': int(self._contains_superlatives(title)),
                'has_emotional_words': int(self._contains_emotional_words(title)),
                'exclamation_count': title.count('!'),
                'colon_count': title.count(':'),
                'dash_count': title.count('-'),
                'pipe_count': title.count('|')
            }
            
            features.append(title_features)
            views.append(row['views_in_period'])
        
        # Calculate correlations
        feature_df = pd.DataFrame(features)
        correlations = {}
        
        for feature in feature_df.columns:
            correlation = np.corrcoef(feature_df[feature], views)[0, 1]
            correlations[feature] = correlation if not np.isnan(correlation) else 0.0
        
        return {
            'title_feature_correlations': correlations,
            'strongest_positive_correlation': max(correlations.items(), key=lambda x: x[1]),
            'strongest_negative_correlation': min(correlations.items(), key=lambda x: x[1])
        }
    
    def generate_insights(self) -> List[str]:
        """Generate actionable insights from the analysis."""
        insights = []
        
        # Channel insights
        channel_performance = {}
        for channel_id in self.data['channel_id'].unique():
            channel_data = self.data[self.data['channel_id'] == channel_id]
            channel_performance[channel_id] = channel_data['views_in_period'].mean()
        
        best_channel = max(channel_performance.items(), key=lambda x: x[1])
        worst_channel = min(channel_performance.items(), key=lambda x: x[1])
        
        insights.append(
            f"Performance varies significantly by channel: "
            f"{best_channel[0]} averages {best_channel[1]:.0f} views while "
            f"{worst_channel[0]} averages {worst_channel[1]:.0f} views"
        )
        
        # Title length insights
        title_lengths = [len(title.split()) for title in self.data['title']]
        high_performers = self.data[self.data['views_in_period'] > self.data['views_in_period'].quantile(0.8)]
        high_perf_lengths = [len(title.split()) for title in high_performers['title']]
        
        if high_perf_lengths:
            avg_length_all = np.mean(title_lengths)
            avg_length_high = np.mean(high_perf_lengths)
            
            if avg_length_high > avg_length_all + 1:
                insights.append(f"High-performing videos tend to have longer titles ({avg_length_high:.1f} vs {avg_length_all:.1f} words)")
            elif avg_length_high < avg_length_all - 1:
                insights.append(f"High-performing videos tend to have shorter titles ({avg_length_high:.1f} vs {avg_length_all:.1f} words)")
        
        # View distribution insights
        views = self.data['views_in_period']
        if views.std() / views.mean() > 1:  # High coefficient of variation
            insights.append("View performance is highly variable - success factors may be channel-specific")
        
        # Pattern insights
        question_titles = self.data[self.data['title'].str.contains(r'\?|^(Why|How|What)', case=False, na=False)]
        if len(question_titles) > 0:
            question_avg_views = question_titles['views_in_period'].mean()
            overall_avg_views = self.data['views_in_period'].mean()
            
            if question_avg_views > overall_avg_views * 1.2:
                insights.append("Question-format titles significantly outperform average")
            elif question_avg_views < overall_avg_views * 0.8:
                insights.append("Question-format titles underperform compared to average")
        
        return insights
    
    # Helper methods for pattern detection
    def _is_question(self, title: str) -> bool:
        """Check if title is a question."""
        return ('?' in title or 
                re.match(r'^(why|how|what|when|where|who|which|can|is|are|do|does|did|will|would|should)', 
                        title.lower()))
    
    def _contains_numbers(self, title: str) -> bool:
        """Check if title contains numbers."""
        return bool(re.search(r'\d', title))
    
    def _contains_superlatives(self, title: str) -> bool:
        """Check if title contains superlative words."""
        superlatives = ['best', 'worst', 'most', 'least', 'biggest', 'smallest', 'fastest', 'slowest', 
                       'greatest', 'ultimate', 'top', 'bottom', 'first', 'last', 'perfect', 'incredible']
        return any(word in title.lower() for word in superlatives)
    
    def _contains_emotional_words(self, title: str) -> bool:
        """Check if title contains emotional trigger words."""
        emotional_words = ['shocking', 'amazing', 'incredible', 'unbelievable', 'secret', 'hidden', 
                          'revealed', 'exposed', 'never', 'always', 'everyone', 'nobody', 'insane',
                          'crazy', 'mind-blowing', 'epic', 'legendary']
        return any(word in title.lower() for word in emotional_words)
    
    def _analyze_punctuation(self, titles: List[str]) -> Dict[str, Any]:
        """Analyze punctuation usage patterns."""
        punctuation_counts = {
            'exclamation': sum(title.count('!') for title in titles),
            'question': sum(title.count('?') for title in titles),
            'colon': sum(title.count(':') for title in titles),
            'dash': sum(title.count('-') for title in titles),
            'pipe': sum(title.count('|') for title in titles),
            'parentheses': sum(title.count('(') for title in titles),
            'brackets': sum(title.count('[') for title in titles)
        }
        
        total_titles = len(titles)
        punctuation_ratios = {k: v / total_titles for k, v in punctuation_counts.items()}
        
        return {
            'counts': punctuation_counts,
            'ratios': punctuation_ratios,
            'most_common_punctuation': max(punctuation_counts.items(), key=lambda x: x[1])[0]
        }