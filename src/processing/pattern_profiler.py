"""
Enhanced Pattern Profiler for TitleCraft AI
Advanced pattern analysis beyond basic data module capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import re
import json
import logging
from datetime import datetime
from dataclasses import dataclass

from ..config import get_config
from ..data.models import ChannelProfile, ChannelStats, TitlePatterns, VideoData

logger = logging.getLogger(__name__)

@dataclass
class StructuralPattern:
    """Represents a structural pattern in titles"""
    pattern: str
    regex: str
    examples: List[str]
    frequency: int
    avg_performance: float
    description: str

@dataclass
class PatternCorrelation:
    """Represents correlation between patterns and performance"""
    pattern_name: str
    correlation_score: float
    high_performer_usage: float
    overall_usage: float
    recommendation: str

class EnhancedPatternProfiler:
    """
    Enhanced pattern profiler with advanced analysis capabilities.
    
    This extends the basic ChannelProfiler with:
    - Structural pattern analysis
    - Performance correlation analysis  
    - Template extraction
    - Emotional trigger analysis
    - Multi-dimensional pattern scoring
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Enhanced pattern definitions
        self.structural_patterns = self._define_structural_patterns()
        self.emotional_triggers = self._define_emotional_triggers()
        self.performance_indicators = self._define_performance_indicators()
        
        # Analysis cache
        self._pattern_cache = {}
    
    def _define_structural_patterns(self) -> Dict[str, StructuralPattern]:
        """Define structural patterns to analyze"""
        patterns = {
            'numbered_list': StructuralPattern(
                pattern="Numbered List",
                regex=r'^\d+\s+\w+',
                examples=["5 Amazing Facts", "10 Best Ways"],
                frequency=0,
                avg_performance=0.0,
                description="Titles starting with numbers (lists, rankings)"
            ),
            'how_to': StructuralPattern(
                pattern="How-To Format", 
                regex=r'^(How\s+to|How\s+\w+)',
                examples=["How to Build", "How NASA Works"],
                frequency=0,
                avg_performance=0.0,
                description="Instructional/tutorial format titles"
            ),
            'question_format': StructuralPattern(
                pattern="Question Format",
                regex=r'^(Why|What|When|Where|Who|Which|Can|Is|Are|Do|Does|Will|Would)\s+',
                examples=["Why Do Tanks", "What Happens When"],
                frequency=0,
                avg_performance=0.0,
                description="Question-style titles"
            ),
            'comparison': StructuralPattern(
                pattern="Comparison Format",
                regex=r'\w+\s+(vs|versus|against|compared\s+to)\s+\w+',
                examples=["Tank vs Tank", "Real vs LEGO"],
                frequency=0,
                avg_performance=0.0,
                description="Comparison between items"
            ),
            'superlative': StructuralPattern(
                pattern="Superlative Format",
                regex=r'(Best|Worst|Most|Least|Biggest|Smallest|Greatest|Ultimate|Top|Bottom)\s+',
                examples=["Best Tank Ever", "Most Amazing Discovery"],
                frequency=0,
                avg_performance=0.0,
                description="Superlative descriptors"
            ),
            'time_reference': StructuralPattern(
                pattern="Time Reference",
                regex=r'(Never|Always|Ever|Today|Now|Finally|Still|Yet)\s+',
                examples=["Never Seen Before", "Finally Revealed"],
                frequency=0,
                avg_performance=0.0,
                description="Time-based urgency or permanence"
            ),
            'insider_knowledge': StructuralPattern(
                pattern="Insider Knowledge",
                regex=r'(Secret|Hidden|Behind|Inside|Real|Truth|Revealed|Exposed)\s+',
                examples=["Secret Behind", "Hidden Truth"],
                frequency=0,
                avg_performance=0.0,
                description="Exclusive or insider information"
            ),
            'emotional_intensity': StructuralPattern(
                pattern="Emotional Intensity",
                regex=r'(Shocking|Amazing|Incredible|Unbelievable|Insane|Crazy|Epic|Mind.?[Bb]lowing)\s+',
                examples=["Shocking Discovery", "Mind-Blowing Facts"],
                frequency=0,
                avg_performance=0.0,
                description="High emotional impact words"
            )
        }
        
        return patterns
    
    def _define_emotional_triggers(self) -> Dict[str, List[str]]:
        """Define emotional trigger categories"""
        return {
            'curiosity': ['secret', 'hidden', 'mystery', 'unknown', 'revealed', 'truth', 'behind'],
            'urgency': ['now', 'today', 'finally', 'never', 'always', 'immediately', 'urgent'],
            'exclusivity': ['only', 'exclusive', 'rare', 'unique', 'special', 'limited', 'private'],
            'surprise': ['shocking', 'unexpected', 'surprising', 'unbelievable', 'incredible', 'amazing'],
            'intensity': ['extreme', 'ultimate', 'massive', 'huge', 'epic', 'legendary', 'insane'],
            'achievement': ['best', 'greatest', 'perfect', 'ultimate', 'champion', 'winner', 'success'],
            'fear': ['dangerous', 'deadly', 'scary', 'terrifying', 'nightmare', 'disaster', 'warning'],
            'social_proof': ['everyone', 'millions', 'thousands', 'popular', 'trending', 'viral', 'famous']
        }
    
    def _define_performance_indicators(self) -> Dict[str, Any]:
        """Define what indicates high performance"""
        return {
            'view_thresholds': {
                'high': 0.8,  # Top 20%
                'medium': 0.5,  # Middle 30%
                'low': 0.2  # Bottom 50%
            },
            'engagement_factors': [
                'click_through_rate',
                'watch_time',
                'likes_ratio', 
                'comment_rate'
            ]
        }
    
    def create_enhanced_profile(self, 
                               channel_data: pd.DataFrame,
                               channel_id: str) -> Dict[str, Any]:
        """
        Create enhanced channel profile with advanced pattern analysis
        
        Args:
            channel_data: DataFrame with channel's video data
            channel_id: Channel identifier
            
        Returns:
            Enhanced profile dictionary with detailed pattern analysis
        """
        logger.info(f"Creating enhanced profile for channel {channel_id}")
        
        if len(channel_data) < self.config.data.min_examples_for_profile:
            logger.warning(f"Channel {channel_id} has only {len(channel_data)} videos, minimum {self.config.data.min_examples_for_profile} recommended")
        
        # Basic metrics
        basic_profile = self._calculate_basic_metrics(channel_data)
        
        # Enhanced pattern analysis
        structural_analysis = self._analyze_structural_patterns(channel_data)
        emotional_analysis = self._analyze_emotional_triggers(channel_data)
        punctuation_analysis = self._analyze_punctuation_patterns(channel_data)
        superlative_analysis = self._analyze_superlative_patterns(channel_data)
        template_analysis = self._analyze_title_templates(channel_data)
        performance_correlations = self._calculate_pattern_correlations(channel_data)
        
        # Template extraction
        successful_templates = self._extract_successful_templates(channel_data)
        
        # Content themes
        content_themes = self._analyze_content_themes(channel_data)
        
        # Performance insights
        performance_insights = self._generate_performance_insights(
            channel_data, structural_analysis, emotional_analysis
        )
        
        enhanced_profile = {
            **basic_profile,
            'structural_patterns': structural_analysis,
            'emotional_triggers': emotional_analysis,
            'punctuation_patterns': punctuation_analysis,
            'superlative_patterns': superlative_analysis,
            'title_templates': template_analysis,
            'pattern_correlations': performance_correlations,
            'successful_templates': successful_templates,
            'content_themes': content_themes,
            'performance_insights': performance_insights,
            'profile_metadata': {
                'created_at': datetime.now().isoformat(),
                'channel_id': channel_id,
                'analysis_version': '2.0',
                'video_count': len(channel_data),
                'data_quality_score': self._calculate_data_quality_score(channel_data)
            }
        }
        
        return enhanced_profile
    
    def _calculate_basic_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic channel metrics"""
        views = data['views_in_period']
        titles = data['title'].tolist()
        
        return {
            'basic_stats': {
                'video_count': len(data),
                'avg_views': float(views.mean()),
                'median_views': float(views.median()),
                'min_views': int(views.min()),
                'max_views': int(views.max()),
                'std_views': float(views.std()),
                'total_views': int(views.sum())
            },
            'title_basics': {
                'avg_length_words': np.mean([len(title.split()) for title in titles]),
                'avg_length_chars': np.mean([len(title) for title in titles]),
                'std_length_words': np.std([len(title.split()) for title in titles]),
                'length_range': {
                    'min_words': min([len(title.split()) for title in titles]),
                    'max_words': max([len(title.split()) for title in titles])
                }
            },
            'performance_thresholds': {
                'high_performer_threshold': float(views.quantile(0.8)),
                'medium_performer_threshold': float(views.quantile(0.5)),
                'low_performer_threshold': float(views.quantile(0.2))
            }
        }
    
    def _analyze_structural_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze structural patterns in titles"""
        titles = data['title'].tolist()
        views = data['views_in_period'].tolist()
        
        pattern_results = {}
        
        for pattern_name, pattern_def in self.structural_patterns.items():
            matches = []
            match_views = []
            
            for i, title in enumerate(titles):
                if re.search(pattern_def.regex, title, re.IGNORECASE):
                    matches.append(title)
                    match_views.append(views[i])
            
            # Calculate statistics
            usage_rate = len(matches) / len(titles) if titles else 0
            avg_performance = np.mean(match_views) if match_views else 0
            
            pattern_results[pattern_name] = {
                'usage_rate': usage_rate,
                'frequency': len(matches),
                'avg_performance': avg_performance,
                'examples': matches[:3],  # Top 3 examples
                'performance_vs_average': avg_performance / np.mean(views) if np.mean(views) > 0 else 0,
                'description': pattern_def.description
            }
        
        return pattern_results
    
    def _analyze_punctuation_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze punctuation usage patterns"""
        titles = data['title'].tolist()
        views = data['views_in_period'].tolist()
        
        punctuation_stats = {}
        punctuation_marks = ['!', '?', ':', ';', '-', '(', ')', '[', ']', '...', '|']
        
        for punct in punctuation_marks:
            punct_titles = []
            punct_views = []
            
            for i, title in enumerate(titles):
                if punct in title:
                    punct_titles.append(title)
                    punct_views.append(views[i])
            
            if punct_titles:
                usage_rate = len(punct_titles) / len(titles)
                avg_performance = np.mean(punct_views)
                
                punctuation_stats[punct] = {
                    'usage_rate': usage_rate,
                    'frequency': len(punct_titles),
                    'avg_performance': avg_performance,
                    'performance_vs_average': avg_performance / np.mean(views) if np.mean(views) > 0 else 0,
                    'examples': punct_titles[:2]
                }
        
        return punctuation_stats
    
    def _analyze_superlative_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze superlative usage patterns"""
        titles = data['title'].tolist()
        views = data['views_in_period'].tolist()
        
        superlatives = {
            'positive': ['best', 'greatest', 'top', 'ultimate', 'perfect', 'amazing', 'incredible'],
            'negative': ['worst', 'terrible', 'awful', 'bad', 'failed', 'disaster'],
            'size': ['biggest', 'largest', 'massive', 'huge', 'tiny', 'smallest'],
            'intensity': ['most', 'extremely', 'super', 'ultra', 'mega', 'epic'],
            'uniqueness': ['only', 'unique', 'rare', 'exclusive', 'special', 'one']
        }
        
        superlative_results = {}
        
        for category, words in superlatives.items():
            matches = []
            match_views = []
            
            for i, title in enumerate(titles):
                title_lower = title.lower()
                if any(word in title_lower for word in words):
                    matches.append(title)
                    match_views.append(views[i])
            
            if matches:
                usage_rate = len(matches) / len(titles)
                avg_performance = np.mean(match_views)
                
                superlative_results[category] = {
                    'usage_rate': usage_rate,
                    'frequency': len(matches),
                    'avg_performance': avg_performance,
                    'performance_vs_average': avg_performance / np.mean(views) if np.mean(views) > 0 else 0,
                    'top_words': [w for w in words if any(w in t.lower() for t in matches)],
                    'examples': matches[:2]
                }
        
        return superlative_results
    
    def _analyze_title_templates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract and analyze title templates"""
        high_performers = data[data['views_in_period'] >= data['views_in_period'].quantile(0.8)]
        titles = high_performers['title'].tolist()
        
        # Template extraction patterns
        template_patterns = {
            'number_x_thing': r'^(\d+)\s+(.+?)(?:\s+(?:that|you|for|to|in|of|with)\s+.+)?$',
            'how_to_x': r'^How\s+to\s+(.+?)(?:\s+(?:in|with|for|using|without)\s+.+)?$',
            'why_x_is_y': r'^Why\s+(.+?)\s+(?:is|are|was|were)\s+(.+?)$',
            'what_happens_when': r'^What\s+(?:happens|would happen)\s+(?:if|when)\s+(.+?)$',
            'x_vs_y': r'^(.+?)\s+(?:vs|versus|against)\s+(.+?)$',
            'the_truth_about': r'^(?:The\s+)?(?:Truth|Reality|Facts?)\s+(?:about|behind|of)\s+(.+?)$',
            'x_that_will_y': r'^(.+?)\s+(?:that|which)\s+will\s+(.+?)$'
        }
        
        template_results = {}
        
        for template_name, pattern in template_patterns.items():
            matches = []
            
            for title in titles:
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    matches.append({
                        'title': title,
                        'template_parts': match.groups()
                    })
            
            if matches:
                template_results[template_name] = {
                    'frequency': len(matches),
                    'usage_rate': len(matches) / len(titles) if titles else 0,
                    'examples': [m['title'] for m in matches[:3]],
                    'template_structure': template_patterns[template_name]
                }
        
        return template_results
    
    def _analyze_emotional_triggers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze emotional triggers in titles"""
        titles = data['title'].tolist()
        views = data['views_in_period'].tolist()
        
        trigger_results = {}
        
        for trigger_category, trigger_words in self.emotional_triggers.items():
            matches = []
            match_views = []
            
            for i, title in enumerate(titles):
                title_lower = title.lower()
                if any(word in title_lower for word in trigger_words):
                    matches.append(title)
                    match_views.append(views[i])
            
            usage_rate = len(matches) / len(titles) if titles else 0
            avg_performance = np.mean(match_views) if match_views else 0
            
            trigger_results[trigger_category] = {
                'usage_rate': usage_rate,
                'frequency': len(matches),
                'avg_performance': avg_performance,
                'examples': matches[:3],
                'trigger_words_found': list(set([
                    word for word in trigger_words 
                    for title in titles 
                    if word in title.lower()
                ]))
            }
        
        return trigger_results
    
    def _calculate_pattern_correlations(self, data: pd.DataFrame) -> List[PatternCorrelation]:
        """Calculate correlations between patterns and performance"""
        high_threshold = data['views_in_period'].quantile(0.8)
        high_performers = data[data['views_in_period'] >= high_threshold]
        
        correlations = []
        
        # Analyze each structural pattern
        for pattern_name, pattern_def in self.structural_patterns.items():
            # Usage in high performers
            high_perf_matches = sum(1 for title in high_performers['title'] 
                                  if re.search(pattern_def.regex, title, re.IGNORECASE))
            high_perf_usage = high_perf_matches / len(high_performers) if len(high_performers) > 0 else 0
            
            # Overall usage
            overall_matches = sum(1 for title in data['title']
                                if re.search(pattern_def.regex, title, re.IGNORECASE))
            overall_usage = overall_matches / len(data) if len(data) > 0 else 0
            
            # Calculate correlation score
            if overall_usage > 0:
                correlation_score = high_perf_usage / overall_usage
            else:
                correlation_score = 0
            
            # Generate recommendation
            recommendation = self._generate_pattern_recommendation(
                pattern_name, correlation_score, high_perf_usage, overall_usage
            )
            
            correlations.append(PatternCorrelation(
                pattern_name=pattern_name,
                correlation_score=correlation_score,
                high_performer_usage=high_perf_usage,
                overall_usage=overall_usage,
                recommendation=recommendation
            ))
        
        # Sort by correlation score
        correlations.sort(key=lambda x: x.correlation_score, reverse=True)
        
        return [c.__dict__ for c in correlations]
    
    def _extract_successful_templates(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract templates from high-performing videos"""
        high_threshold = data['views_in_period'].quantile(0.8)
        high_performers = data[data['views_in_period'] >= high_threshold]
        
        templates = []
        
        # Extract common patterns from high performers
        high_perf_titles = high_performers['title'].tolist()
        
        # Simple template extraction (can be enhanced with NLP)
        for title in high_perf_titles[:10]:  # Top 10 performers
            # Generate template by replacing specific words with placeholders
            template = self._generalize_title(title)
            
            templates.append({
                'template': template,
                'original_title': title,
                'views': int(high_performers[high_performers['title'] == title]['views_in_period'].iloc[0]),
                'confidence': self._calculate_template_confidence(template, high_perf_titles)
            })
        
        # Remove duplicates and sort by confidence
        unique_templates = []
        seen_templates = set()
        
        for template in templates:
            if template['template'] not in seen_templates:
                unique_templates.append(template)
                seen_templates.add(template['template'])
        
        return sorted(unique_templates, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _analyze_content_themes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content themes and topics"""
        titles = data['title'].tolist()
        summaries = data['summary'].fillna('').tolist()
        
        # Combine titles and summaries for theme analysis
        all_text = ' '.join(titles + summaries).lower()
        
        # Simple keyword extraction (can be enhanced with TF-IDF or topic modeling)
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in stop_words and len(word) > 2}
        
        # Identify content categories
        content_categories = self._categorize_content(titles, summaries)
        
        return {
            'top_keywords': dict(Counter(filtered_words).most_common(20)),
            'content_categories': content_categories,
            'themes_summary': self._summarize_themes(filtered_words, content_categories)
        }
    
    def _generate_performance_insights(self, 
                                     data: pd.DataFrame,
                                     structural_analysis: Dict[str, Any],
                                     emotional_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable performance insights"""
        
        # Find top performing patterns
        top_structural_patterns = sorted(
            [(name, info['performance_vs_average']) for name, info in structural_analysis.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        top_emotional_triggers = sorted(
            [(name, info['avg_performance']) for name, info in emotional_analysis.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        # Calculate optimal title length
        title_lengths = [len(title.split()) for title in data['title']]
        views = data['views_in_period'].tolist()
        
        length_performance = defaultdict(list)
        for length, view_count in zip(title_lengths, views):
            length_performance[length].append(view_count)
        
        avg_performance_by_length = {
            length: np.mean(views) 
            for length, views in length_performance.items()
        }
        
        optimal_length = max(avg_performance_by_length.items(), 
                           key=lambda x: x[1])[0] if avg_performance_by_length else 7
        
        insights = {
            'top_performing_patterns': [
                {'pattern': name, 'performance_multiplier': score}
                for name, score in top_structural_patterns
            ],
            'most_effective_emotional_triggers': [
                {'trigger': name, 'avg_views': score}
                for name, score in top_emotional_triggers
            ],
            'optimal_title_length': optimal_length,
            'length_performance_analysis': avg_performance_by_length,
            'recommendations': self._generate_recommendations(
                top_structural_patterns, top_emotional_triggers, optimal_length
            )
        }
        
        return insights
    
    def _generate_pattern_recommendation(self, 
                                       pattern_name: str,
                                       correlation_score: float,
                                       high_perf_usage: float,
                                       overall_usage: float) -> str:
        """Generate recommendation for pattern usage"""
        
        if correlation_score > 1.2:
            return f"Highly recommended: {pattern_name} shows strong performance correlation. Use more frequently."
        elif correlation_score > 0.8:
            return f"Recommended: {pattern_name} shows good performance. Consider using more."
        elif correlation_score < 0.5:
            return f"Avoid: {pattern_name} shows poor performance correlation. Use sparingly."
        else:
            return f"Neutral: {pattern_name} shows average performance. Use as needed."
    
    def _generalize_title(self, title: str) -> str:
        """Convert specific title to general template"""
        # Simple generalization - replace numbers and specific words
        template = re.sub(r'\d+', '[NUMBER]', title)
        
        # Replace specific nouns with placeholders (simplified)
        specific_words = ['tank', 'lego', 'space', 'nasa', 'solar', 'planet']
        for word in specific_words:
            template = re.sub(word, '[TOPIC]', template, flags=re.IGNORECASE)
        
        return template
    
    def _calculate_template_confidence(self, template: str, titles: List[str]) -> float:
        """Calculate confidence score for template"""
        # Simple confidence based on how many titles match the pattern
        matches = sum(1 for title in titles if self._matches_template(title, template))
        return matches / len(titles) if titles else 0
    
    def _matches_template(self, title: str, template: str) -> bool:
        """Check if title matches template pattern"""
        # Convert template to regex pattern
        pattern = template.replace('[NUMBER]', r'\d+').replace('[TOPIC]', r'\w+')
        return bool(re.search(pattern, title, re.IGNORECASE))
    
    def _categorize_content(self, titles: List[str], summaries: List[str]) -> Dict[str, float]:
        """Categorize content into themes"""
        all_text = ' '.join(titles + summaries).lower()
        
        categories = {
            'military/war': ['war', 'military', 'tank', 'weapon', 'battle', 'army', 'combat'],
            'space/science': ['space', 'planet', 'nasa', 'universe', 'solar', 'science'],
            'gaming/toys': ['lego', 'game', 'toy', 'build', 'play', 'minecraft'],
            'education': ['how', 'learn', 'explain', 'tutorial', 'guide', 'facts'],
            'entertainment': ['funny', 'comedy', 'show', 'viral', 'epic', 'amazing']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(all_text.count(keyword) for keyword in keywords)
            category_scores[category] = score / len(all_text) if all_text else 0
        
        return category_scores
    
    def _summarize_themes(self, keywords: Dict[str, int], categories: Dict[str, float]) -> str:
        """Generate human-readable theme summary"""
        top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "general"
        top_keywords = list(Counter(keywords).most_common(5))
        
        return f"Primary theme: {top_category}. Key topics: {', '.join([kw for kw, _ in top_keywords])}"
    
    def _generate_recommendations(self, 
                                top_patterns: List[Tuple[str, float]],
                                top_triggers: List[Tuple[str, float]],
                                optimal_length: int) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Length recommendation
        recommendations.append(f"Aim for {optimal_length} words in titles for optimal performance")
        
        # Pattern recommendations
        if top_patterns:
            best_pattern = top_patterns[0][0]
            recommendations.append(f"Use '{best_pattern}' pattern more frequently - shows {top_patterns[0][1]:.1f}x better performance")
        
        # Trigger recommendations
        if top_triggers:
            best_trigger = top_triggers[0][0]
            recommendations.append(f"Incorporate '{best_trigger}' emotional triggers - highest performing category")
        
        return recommendations
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        # Simple quality metrics
        completeness = 1 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
        
        # Check for reasonable view ranges
        views = data['views_in_period']
        view_validity = 1.0 if views.min() >= 0 and views.max() < 10000000 else 0.8
        
        # Check title quality
        title_quality = 1.0 if all(len(title) > 5 for title in data['title']) else 0.9
        
        return (completeness + view_validity + title_quality) / 3