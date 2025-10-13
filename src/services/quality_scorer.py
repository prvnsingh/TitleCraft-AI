"""
Quality Scoring System for TitleCraft AI

This module provides comprehensive quality evaluation and ranking for generated titles
using multiple scoring dimensions and predictive performance modeling.
"""

import logging
import numpy as np
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter
import math

from ..data.models import ChannelProfile, VideoData
from ..processing.llm_orchestrator import GeneratedTitle

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Comprehensive quality score for a title"""
    overall_score: float
    dimension_scores: Dict[str, float]
    performance_prediction: float
    confidence_level: float
    improvement_suggestions: List[str]
    risk_factors: List[str]
    explanation: str


@dataclass
class ScoringDimension:
    """Definition of a scoring dimension"""
    name: str
    weight: float
    min_score: float
    max_score: float
    description: str
    scorer_function: str


class TitleQualityScorer:
    """
    Advanced quality scoring system for YouTube titles.
    
    Evaluates titles across multiple dimensions:
    - Clickability (emotional appeal, curiosity)
    - Searchability (SEO, keywords)
    - Channel fit (pattern alignment)
    - Performance prediction (view estimation)
    - Risk assessment (policy, brand safety)
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # Initialize scoring dimensions
        self.dimensions = self._initialize_scoring_dimensions()
        
        # Load scoring models
        self.pattern_weights = self._initialize_pattern_weights()
        self.performance_model = self._initialize_performance_model()
        
        # Emotional and power word dictionaries
        self.emotional_words = self._load_emotional_dictionaries()
        self.power_words = self._load_power_words()
        self.negative_indicators = self._load_negative_indicators()
    
    def _initialize_scoring_dimensions(self) -> Dict[str, ScoringDimension]:
        """Initialize all scoring dimensions with weights"""
        return {
            'clickability': ScoringDimension(
                name="Clickability",
                weight=0.25,
                min_score=0.0,
                max_score=1.0,
                description="Emotional appeal and click-through rate potential",
                scorer_function="score_clickability"
            ),
            'searchability': ScoringDimension(
                name="Searchability", 
                weight=0.20,
                min_score=0.0,
                max_score=1.0,
                description="SEO potential and discoverability",
                scorer_function="score_searchability"
            ),
            'channel_fit': ScoringDimension(
                name="Channel Fit",
                weight=0.25,
                min_score=0.0,
                max_score=1.0,
                description="Alignment with channel patterns and brand",
                scorer_function="score_channel_fit"
            ),
            'readability': ScoringDimension(
                name="Readability",
                weight=0.10,
                min_score=0.0,
                max_score=1.0,
                description="Clarity and ease of understanding",
                scorer_function="score_readability"
            ),
            'uniqueness': ScoringDimension(
                name="Uniqueness",
                weight=0.10,
                min_score=0.0,
                max_score=1.0,
                description="Originality and differentiation",
                scorer_function="score_uniqueness"
            ),
            'risk_assessment': ScoringDimension(
                name="Risk Assessment",
                weight=0.10,
                min_score=0.0,
                max_score=1.0,
                description="Policy compliance and brand safety",
                scorer_function="score_risk_assessment"
            )
        }
    
    def _initialize_pattern_weights(self) -> Dict[str, float]:
        """Initialize pattern importance weights"""
        return {
            'question_format': 0.15,
            'numbered_list': 0.20,
            'how_to': 0.18,
            'superlative': 0.12,
            'emotional_hook': 0.25,
            'comparison': 0.10
        }
    
    def _initialize_performance_model(self) -> Dict[str, Any]:
        """Initialize performance prediction model parameters"""
        return {
            'base_multipliers': {
                'emotional_words': 1.2,
                'power_words': 1.15,
                'question_format': 1.1,
                'numbered_list': 1.25,
                'superlative': 1.08,
                'length_optimal': 1.1
            },
            'penalty_factors': {
                'too_long': 0.9,
                'too_short': 0.85,
                'low_emotional_appeal': 0.8,
                'poor_readability': 0.7
            }
        }
    
    def _load_emotional_dictionaries(self) -> Dict[str, List[str]]:
        """Load emotional word dictionaries by category"""
        return {
            'curiosity': [
                'secret', 'hidden', 'mystery', 'revealed', 'truth', 'behind', 'inside',
                'unknown', 'discovered', 'exposed', 'leaked', 'exclusive'
            ],
            'urgency': [
                'now', 'today', 'urgent', 'immediate', 'breaking', 'just', 'finally',
                'never', 'always', 'last chance', 'limited time'
            ],
            'surprise': [
                'shocking', 'amazing', 'incredible', 'unbelievable', 'mind-blowing',
                'unexpected', 'surprising', 'stunning', 'astonishing'
            ],
            'fear': [
                'dangerous', 'deadly', 'scary', 'terrifying', 'warning', 'alert',
                'disaster', 'catastrophe', 'nightmare', 'threat'
            ],
            'achievement': [
                'best', 'greatest', 'ultimate', 'perfect', 'champion', 'winner',
                'success', 'triumph', 'victory', 'breakthrough'
            ]
        }
    
    def _load_power_words(self) -> List[str]:
        """Load high-impact power words"""
        return [
            'ultimate', 'essential', 'crucial', 'vital', 'critical', 'proven',
            'guaranteed', 'instant', 'effortless', 'powerful', 'advanced',
            'professional', 'expert', 'master', 'complete', 'comprehensive'
        ]
    
    def _load_negative_indicators(self) -> Dict[str, List[str]]:
        """Load negative indicators for risk assessment"""
        return {
            'clickbait_red_flags': [
                'doctors hate', 'you won\'t believe', 'this will shock you',
                'number 7 will', 'what happens next', 'gone wrong'
            ],
            'policy_risks': [
                'illegal', 'banned', 'prohibited', 'dangerous', 'hack',
                'cheat', 'exploit', 'scam', 'fraud'
            ],
            'low_quality_indicators': [
                'random', 'weird', 'wtf', 'omg', 'lol', 'epic fail',
                'so random', 'totally'
            ]
        }
    
    def score_title(self, 
                   title: str,
                   channel_profile: ChannelProfile,
                   similar_examples: List[VideoData] = None,
                   enhanced_profile: Dict[str, Any] = None) -> QualityScore:
        """
        Score a title across all quality dimensions
        
        Args:
            title: Title to score
            channel_profile: Channel profile data
            similar_examples: Similar successful titles
            enhanced_profile: Enhanced pattern analysis
            
        Returns:
            Comprehensive quality score
        """
        
        # Calculate scores for each dimension
        dimension_scores = {}
        improvement_suggestions = []
        risk_factors = []
        
        for dim_name, dimension in self.dimensions.items():
            scorer_method = getattr(self, dimension.scorer_function)
            score = scorer_method(title, channel_profile, similar_examples, enhanced_profile)
            dimension_scores[dim_name] = score
            
            # Collect improvement suggestions
            if score < 0.7:
                suggestions = self._get_improvement_suggestions(dim_name, score, title)
                improvement_suggestions.extend(suggestions)
            
            # Collect risk factors
            if dim_name == 'risk_assessment' and score < 0.8:
                risks = self._identify_risk_factors(title)
                risk_factors.extend(risks)
        
        # Calculate weighted overall score
        overall_score = sum(
            score * self.dimensions[dim_name].weight 
            for dim_name, score in dimension_scores.items()
        )
        
        # Predict performance
        performance_prediction = self._predict_performance(
            title, channel_profile, dimension_scores, enhanced_profile
        )
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence(
            dimension_scores, channel_profile, enhanced_profile
        )
        
        # Generate explanation
        explanation = self._generate_score_explanation(
            overall_score, dimension_scores, performance_prediction
        )
        
        return QualityScore(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            performance_prediction=performance_prediction,
            confidence_level=confidence_level,
            improvement_suggestions=improvement_suggestions,
            risk_factors=risk_factors,
            explanation=explanation
        )
    
    def score_clickability(self, 
                          title: str,
                          channel_profile: ChannelProfile,
                          similar_examples: List[VideoData] = None,
                          enhanced_profile: Dict[str, Any] = None) -> float:
        """Score title clickability and emotional appeal"""
        
        score = 0.0
        title_lower = title.lower()
        
        # Emotional word scoring
        emotional_score = 0.0
        for emotion_category, words in self.emotional_words.items():
            matches = sum(1 for word in words if word in title_lower)
            if matches > 0:
                emotional_score += min(matches * 0.2, 0.4)  # Cap at 0.4 per category
        
        score += min(emotional_score, 0.3)  # Cap total emotional score at 0.3
        
        # Power words
        power_word_count = sum(1 for word in self.power_words if word in title_lower)
        score += min(power_word_count * 0.1, 0.2)
        
        # Curiosity gap indicators
        curiosity_patterns = [
            r'why\\s+\\w+',
            r'what\\s+happens',
            r'\\d+\\s+things',
            r'secret\\s+\\w+',
            r'truth\\s+about',
            r'behind\\s+the'
        ]
        
        curiosity_score = sum(0.1 for pattern in curiosity_patterns 
                             if re.search(pattern, title_lower))
        score += min(curiosity_score, 0.2)
        
        # Numbers and lists (inherently clickable)
        if re.search(r'^\\d+', title):
            score += 0.15
        
        # Question format
        if title.strip().endswith('?'):
            score += 0.1
        
        # Superlatives
        superlatives = ['best', 'worst', 'top', 'ultimate', 'greatest']
        if any(sup in title_lower for sup in superlatives):
            score += 0.1
        
        return min(score, 1.0)
    
    def score_searchability(self,
                           title: str,
                           channel_profile: ChannelProfile, 
                           similar_examples: List[VideoData] = None,
                           enhanced_profile: Dict[str, Any] = None) -> float:
        """Score SEO potential and discoverability"""
        
        score = 0.0
        
        # Length optimization (YouTube optimal is 60-70 characters)
        title_length = len(title)
        if 50 <= title_length <= 70:
            score += 0.25
        elif 40 <= title_length <= 80:
            score += 0.15
        else:
            score += max(0, 0.1 - abs(title_length - 60) * 0.01)
        
        # Keyword density (avoid over-stuffing)
        words = title.lower().split()
        word_count = len(words)
        unique_words = len(set(words))
        
        if word_count > 0:
            keyword_density = unique_words / word_count
            if keyword_density >= 0.8:  # Good keyword diversity
                score += 0.2
            else:
                score += keyword_density * 0.25
        
        # Common search patterns
        search_patterns = [
            'how to', 'what is', 'best', 'review', 'vs', 'tutorial',
            'guide', 'tips', 'tricks', 'explained'
        ]
        
        pattern_matches = sum(1 for pattern in search_patterns 
                             if pattern in title.lower())
        score += min(pattern_matches * 0.1, 0.2)
        
        # Avoid stop word heavy titles
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        stop_word_ratio = sum(1 for word in words if word in stop_words) / max(word_count, 1)
        if stop_word_ratio < 0.3:
            score += 0.15
        
        # Readability (simple words score better for search)
        complex_word_count = sum(1 for word in words if len(word) > 8)
        if complex_word_count / max(word_count, 1) < 0.2:
            score += 0.1
        
        return min(score, 1.0)
    
    def score_channel_fit(self,
                         title: str,
                         channel_profile: ChannelProfile,
                         similar_examples: List[VideoData] = None,
                         enhanced_profile: Dict[str, Any] = None) -> float:
        """Score alignment with channel patterns"""
        
        score = 0.0
        
        # Length alignment
        title_word_count = len(title.split())
        optimal_length = channel_profile.title_patterns.avg_length_words
        length_diff = abs(title_word_count - optimal_length)
        
        if length_diff <= 1:
            score += 0.2
        elif length_diff <= 2:
            score += 0.15
        else:
            score += max(0, 0.1 - length_diff * 0.02)
        
        # Pattern alignment
        patterns = channel_profile.title_patterns
        
        # Question format alignment
        is_question = title.strip().endswith('?')
        if is_question and patterns.question_ratio > 0.2:
            score += 0.15
        elif not is_question and patterns.question_ratio <= 0.2:
            score += 0.1
        
        # Numeric content alignment
        has_numbers = bool(re.search(r'\\d+', title))
        if has_numbers and patterns.numeric_ratio > 0.3:
            score += 0.15
        elif not has_numbers and patterns.numeric_ratio <= 0.3:
            score += 0.1
        
        # Superlative alignment
        superlatives = ['best', 'worst', 'top', 'ultimate', 'greatest', 'most', 'least']
        has_superlative = any(sup in title.lower() for sup in superlatives)
        if has_superlative and patterns.superlative_ratio > 0.2:
            score += 0.15
        elif not has_superlative and patterns.superlative_ratio <= 0.2:
            score += 0.1
        
        # Emotional hook alignment
        emotional_words_found = sum(
            1 for emotion_words in self.emotional_words.values()
            for word in emotion_words
            if word in title.lower()
        )
        
        if emotional_words_found > 0 and patterns.emotional_hook_ratio > 0.3:
            score += 0.2
        elif emotional_words_found == 0 and patterns.emotional_hook_ratio <= 0.3:
            score += 0.15
        
        # Enhanced pattern alignment if available
        if enhanced_profile and 'structural_patterns' in enhanced_profile:
            structural_score = self._score_structural_alignment(title, enhanced_profile)
            score += structural_score * 0.25
        
        return min(score, 1.0)
    
    def score_readability(self,
                         title: str,
                         channel_profile: ChannelProfile,
                         similar_examples: List[VideoData] = None,
                         enhanced_profile: Dict[str, Any] = None) -> float:
        """Score title readability and clarity"""
        
        score = 0.0
        words = title.split()
        
        # Average word length (shorter words = more readable)
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        if avg_word_length <= 5:
            score += 0.3
        elif avg_word_length <= 7:
            score += 0.2
        else:
            score += max(0, 0.1 - (avg_word_length - 7) * 0.05)
        
        # Sentence structure complexity
        if title.count(',') <= 1 and title.count('(') == 0:
            score += 0.2  # Simple structure
        
        # Capital letters (not ALL CAPS)
        if not title.isupper() and title[0].isupper():
            score += 0.15
        
        # No excessive punctuation
        punct_count = sum(1 for char in title if char in '!?.:;')
        if punct_count <= 2:
            score += 0.15
        
        # Proper spacing and formatting
        if not re.search(r'\\s{2,}', title) and not title.startswith(' '):
            score += 0.1
        
        # Avoid jargon overload
        technical_words = sum(1 for word in words if len(word) > 10)
        if technical_words / max(len(words), 1) < 0.2:
            score += 0.1
        
        return min(score, 1.0)
    
    def score_uniqueness(self,
                        title: str,
                        channel_profile: ChannelProfile,
                        similar_examples: List[VideoData] = None,
                        enhanced_profile: Dict[str, Any] = None) -> float:
        """Score title uniqueness and originality"""
        
        score = 0.0
        
        # Check against similar examples
        if similar_examples:
            title_words = set(title.lower().split())
            
            similarities = []
            for example in similar_examples:
                example_words = set(example.title.lower().split())
                if title_words and example_words:
                    overlap = len(title_words.intersection(example_words))
                    similarity = overlap / len(title_words.union(example_words))
                    similarities.append(similarity)
            
            if similarities:
                max_similarity = max(similarities)
                uniqueness_score = 1.0 - max_similarity
                score += uniqueness_score * 0.5
            else:
                score += 0.5
        else:
            score += 0.3  # Partial credit if no examples to compare
        
        # Avoid generic phrases
        generic_phrases = [
            'ultimate guide', 'complete tutorial', 'everything you need',
            'full review', 'honest review', 'my thoughts'
        ]
        
        generic_count = sum(1 for phrase in generic_phrases 
                           if phrase in title.lower())
        score += max(0, 0.2 - generic_count * 0.1)
        
        # Reward creative word combinations
        words = title.lower().split()
        if len(words) >= 3:
            # Check for interesting adjective-noun combinations
            creative_indicators = ['epic', 'insane', 'mind-blowing', 'genius', 'legendary']
            creative_count = sum(1 for word in creative_indicators if word in words)
            score += min(creative_count * 0.1, 0.2)
        
        # Reward specific details over generic terms
        specific_indicators = re.findall(r'\\d+', title)  # Numbers are often specific
        if specific_indicators:
            score += 0.1
        
        return min(score, 1.0)
    
    def score_risk_assessment(self,
                             title: str,
                             channel_profile: ChannelProfile,
                             similar_examples: List[VideoData] = None,
                             enhanced_profile: Dict[str, Any] = None) -> float:
        """Assess title for policy and brand safety risks"""
        
        score = 1.0  # Start with perfect score, subtract for risks
        title_lower = title.lower()
        
        # Check for clickbait red flags
        for phrase in self.negative_indicators['clickbait_red_flags']:
            if phrase in title_lower:
                score -= 0.3
        
        # Check for policy risks
        for risk_word in self.negative_indicators['policy_risks']:
            if risk_word in title_lower:
                score -= 0.4
        
        # Check for low quality indicators
        for indicator in self.negative_indicators['low_quality_indicators']:
            if indicator in title_lower:
                score -= 0.2
        
        # Excessive capitalization
        if title.isupper():
            score -= 0.3
        
        # Excessive punctuation
        punct_count = sum(1 for char in title if char in '!?')
        if punct_count > 3:
            score -= 0.2
        
        # Check for misleading language
        misleading_terms = ['guaranteed', 'secret doctors', 'one weird trick', 'miracle']
        misleading_count = sum(1 for term in misleading_terms if term in title_lower)
        score -= misleading_count * 0.25
        
        return max(score, 0.0)
    
    def _score_structural_alignment(self, title: str, enhanced_profile: Dict) -> float:
        """Score alignment with enhanced structural patterns"""
        
        structural_patterns = enhanced_profile.get('structural_patterns', {})
        score = 0.0
        
        for pattern_name, pattern_data in structural_patterns.items():
            usage_rate = pattern_data.get('usage_rate', 0)
            performance_ratio = pattern_data.get('performance_vs_average', 1)
            
            # Check if title matches this pattern
            pattern_matches = False
            if pattern_name == 'numbered_list' and re.search(r'^\\d+', title):
                pattern_matches = True
            elif pattern_name == 'how_to' and 'how to' in title.lower():
                pattern_matches = True
            elif pattern_name == 'question_format' and title.strip().endswith('?'):
                pattern_matches = True
            
            if pattern_matches and performance_ratio > 1.1:
                score += usage_rate * performance_ratio * 0.1
        
        return min(score, 1.0)
    
    def _predict_performance(self,
                            title: str,
                            channel_profile: ChannelProfile,
                            dimension_scores: Dict[str, float],
                            enhanced_profile: Dict[str, Any] = None) -> float:
        """Predict relative performance compared to channel average"""
        
        base_performance = channel_profile.stats.avg_views
        multiplier = 1.0
        
        # Apply dimension-based multipliers
        clickability_boost = 1.0 + (dimension_scores['clickability'] - 0.5) * 0.4
        searchability_boost = 1.0 + (dimension_scores['searchability'] - 0.5) * 0.3
        channel_fit_boost = 1.0 + (dimension_scores['channel_fit'] - 0.5) * 0.5
        
        multiplier *= clickability_boost * searchability_boost * channel_fit_boost
        
        # Apply pattern-based multipliers
        title_lower = title.lower()
        
        # Emotional words boost
        emotional_count = sum(
            1 for emotion_words in self.emotional_words.values()
            for word in emotion_words
            if word in title_lower
        )
        if emotional_count > 0:
            multiplier *= self.performance_model['base_multipliers']['emotional_words']
        
        # Power words boost
        power_word_count = sum(1 for word in self.power_words if word in title_lower)
        if power_word_count > 0:
            multiplier *= self.performance_model['base_multipliers']['power_words']
        
        # Length optimization
        title_length = len(title.split())
        optimal_length = channel_profile.title_patterns.avg_length_words
        if abs(title_length - optimal_length) <= 2:
            multiplier *= self.performance_model['base_multipliers']['length_optimal']
        
        # Risk penalties
        if dimension_scores['risk_assessment'] < 0.7:
            multiplier *= 0.8  # Risk penalty
        
        predicted_views = base_performance * multiplier
        
        # Normalize to performance ratio
        return min(predicted_views / base_performance, 3.0)  # Cap at 3x performance
    
    def _calculate_confidence(self,
                             dimension_scores: Dict[str, float],
                             channel_profile: ChannelProfile,
                             enhanced_profile: Dict[str, Any] = None) -> float:
        """Calculate confidence level in the prediction"""
        
        # Base confidence on data quality
        video_count = channel_profile.stats.total_videos
        data_confidence = min(video_count / 100.0, 1.0)
        
        # Confidence based on score consistency
        scores = list(dimension_scores.values())
        score_variance = np.var(scores) if scores else 0
        consistency_confidence = max(0, 1.0 - score_variance * 2)
        
        # Enhanced data improves confidence
        enhanced_confidence = 0.1 if enhanced_profile else 0
        
        overall_confidence = (data_confidence * 0.6 + 
                            consistency_confidence * 0.3 + 
                            enhanced_confidence)
        
        return min(overall_confidence, 1.0)
    
    def _generate_score_explanation(self,
                                   overall_score: float,
                                   dimension_scores: Dict[str, float],
                                   performance_prediction: float) -> str:
        """Generate human-readable explanation of the score"""
        
        # Overall assessment
        if overall_score >= 0.8:
            quality_level = "Excellent"
        elif overall_score >= 0.7:
            quality_level = "Good"
        elif overall_score >= 0.6:
            quality_level = "Average"
        else:
            quality_level = "Needs Improvement"
        
        # Top performing dimensions
        sorted_dimensions = sorted(dimension_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        top_dimension = sorted_dimensions[0][0]
        weakest_dimension = sorted_dimensions[-1][0]
        
        # Performance prediction interpretation
        if performance_prediction >= 1.5:
            performance_level = "significantly above average"
        elif performance_prediction >= 1.2:
            performance_level = "above average"
        elif performance_prediction >= 0.8:
            performance_level = "average"
        else:
            performance_level = "below average"
        
        explanation = f"""{quality_level} title (Score: {overall_score:.1%}).

Strongest aspect: {top_dimension.replace('_', ' ').title()} ({dimension_scores[top_dimension]:.1%})
Improvement area: {weakest_dimension.replace('_', ' ').title()} ({dimension_scores[weakest_dimension]:.1%})

Predicted to perform {performance_level} ({performance_prediction:.1f}x channel average)."""
        
        return explanation
    
    def _get_improvement_suggestions(self, 
                                    dimension: str, 
                                    score: float, 
                                    title: str) -> List[str]:
        """Get specific improvement suggestions for low-scoring dimensions"""
        
        suggestions = []
        
        if dimension == 'clickability' and score < 0.7:
            suggestions.extend([
                "Add emotional trigger words (amazing, shocking, secret)",
                "Include numbers or lists for higher engagement", 
                "Create a curiosity gap with 'Why' or 'What happens when'"
            ])
        
        elif dimension == 'searchability' and score < 0.7:
            suggestions.extend([
                "Optimize length to 50-70 characters",
                "Include searchable keywords naturally",
                "Use common search patterns (How to, Best, Review)"
            ])
        
        elif dimension == 'channel_fit' and score < 0.7:
            suggestions.extend([
                "Match your typical title length more closely",
                "Align with your channel's question/statement format",
                "Use patterns that perform well for your channel"
            ])
        
        elif dimension == 'readability' and score < 0.7:
            suggestions.extend([
                "Simplify complex words",
                "Reduce sentence complexity",
                "Check capitalization and punctuation"
            ])
        
        elif dimension == 'uniqueness' and score < 0.7:
            suggestions.extend([
                "Avoid generic phrases like 'ultimate guide'",
                "Add specific details or numbers",
                "Create unique angle or perspective"
            ])
        
        return suggestions
    
    def _identify_risk_factors(self, title: str) -> List[str]:
        """Identify specific risk factors in the title"""
        
        risks = []
        title_lower = title.lower()
        
        # Check for specific risk patterns
        for phrase in self.negative_indicators['clickbait_red_flags']:
            if phrase in title_lower:
                risks.append(f"Potential clickbait phrase: '{phrase}'")
        
        for word in self.negative_indicators['policy_risks']:
            if word in title_lower:
                risks.append(f"Policy risk keyword: '{word}'")
        
        if title.isupper():
            risks.append("Excessive capitalization may appear unprofessional")
        
        punct_count = sum(1 for char in title if char in '!?')
        if punct_count > 3:
            risks.append("Excessive punctuation may appear spammy")
        
        return risks
    
    def rank_titles(self, 
                   titles: List[str],
                   channel_profile: ChannelProfile,
                   similar_examples: List[VideoData] = None,
                   enhanced_profile: Dict[str, Any] = None) -> List[Tuple[str, QualityScore]]:
        """
        Rank multiple titles by quality score
        
        Returns:
            List of (title, quality_score) tuples sorted by score
        """
        
        scored_titles = []
        
        for title in titles:
            score = self.score_title(title, channel_profile, similar_examples, enhanced_profile)
            scored_titles.append((title, score))
        
        # Sort by overall score (descending)
        scored_titles.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return scored_titles