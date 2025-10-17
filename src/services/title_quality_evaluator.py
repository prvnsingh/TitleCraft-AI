"""
Title Quality Evaluator
Evaluates and ranks generated titles based on discovered patterns
"""

import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np

from src.data_module.data_processor import GeneratedTitle
from .pattern_discovery import IntelligentPatterns


@dataclass
class TitleScore:
    """Comprehensive scoring for a generated title"""
    title: str
    overall_score: float
    confidence_score: float
    pattern_scores: Dict[str, float]
    predicted_performance: str  # "high", "medium", "low"
    reasoning: str
    strengths: List[str]
    recommendations: List[str]


class TitleQualityEvaluator:
    """
    Evaluates and ranks titles based on intelligent pattern analysis
    """
    
    def __init__(self):
        self.score_weights = {
            'word_count': 0.0,  # Will be set from pattern weights
            'question_usage': 0.0,
            'numeric_usage': 0.0,
            'exclamation_usage': 0.0,
            'capitalization': 0.0,
            'keyword_match': 0.0
        }
    
    def evaluate_and_rank_titles(self, titles: List[GeneratedTitle], 
                                patterns: IntelligentPatterns) -> List[TitleScore]:
        """
        Evaluate all titles and return them ranked by predicted performance
        """
        # Update score weights from pattern analysis
        self._update_weights_from_patterns(patterns)
        
        # Score each title
        title_scores = []
        for title_obj in titles:
            score = self._evaluate_single_title(title_obj.title, patterns)
            score.reasoning = title_obj.reasoning  # Keep original reasoning
            title_scores.append(score)
        
        # Rank by overall score
        title_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return title_scores
    
    def _update_weights_from_patterns(self, patterns: IntelligentPatterns):
        """Update scoring weights based on discovered pattern importance"""
        self.score_weights = {
            'word_count': patterns.pattern_weights.word_count_weight,
            'question_usage': patterns.pattern_weights.question_weight,
            'numeric_usage': patterns.pattern_weights.numeric_weight,
            'exclamation_usage': patterns.pattern_weights.exclamation_weight,
            'capitalization': patterns.pattern_weights.capitalization_weight,
            'keyword_match': patterns.pattern_weights.keyword_weight
        }
    
    def _evaluate_single_title(self, title: str, patterns: IntelligentPatterns) -> TitleScore:
        """Evaluate a single title against discovered patterns"""
        
        # Calculate individual pattern scores
        pattern_scores = self._calculate_pattern_scores(title, patterns)
        
        # Calculate weighted overall score
        overall_score = self._calculate_overall_score(pattern_scores)
        
        # Determine confidence based on pattern confidence and title characteristics
        confidence = self._calculate_title_confidence(title, patterns, pattern_scores)
        
        # Predict performance category
        performance = self._predict_performance_category(overall_score, confidence)
        
        # Generate detailed analysis
        strengths = self._identify_strengths(title, pattern_scores, patterns)
        recommendations = self._generate_recommendations(title, pattern_scores, patterns)
        reasoning = self._generate_detailed_reasoning(title, pattern_scores, patterns)
        
        return TitleScore(
            title=title,
            overall_score=overall_score,
            confidence_score=confidence,
            pattern_scores=pattern_scores,
            predicted_performance=performance,
            reasoning=reasoning,
            strengths=strengths,
            recommendations=recommendations
        )
    
    def _calculate_pattern_scores(self, title: str, patterns: IntelligentPatterns) -> Dict[str, float]:
        """Calculate scores for each pattern type"""
        scores = {}
        
        # Word count score
        scores['word_count'] = self._calculate_word_count_score(title, patterns)
        
        # Usage-based scores
        scores['question_usage'] = self._calculate_usage_score('?' in title, patterns.question_percentage, 0.1)
        scores['numeric_usage'] = self._calculate_usage_score(bool(re.search(r'\d+', title)), patterns.numeric_percentage, 0.2)
        scores['exclamation_usage'] = self._calculate_usage_score('!' in title, patterns.exclamation_percentage, 0.1)
        
        # Capitalization score
        scores['capitalization'] = self._calculate_capitalization_score(title, patterns)
        
        # Keyword match score
        scores['keyword_match'] = self._calculate_keyword_score(title, patterns)
        
        return scores
    
    def _calculate_word_count_score(self, title: str, patterns: IntelligentPatterns) -> float:
        """Calculate word count alignment score"""
        word_count = len(title.split())
        target_count = patterns.avg_word_count
        word_diff = abs(word_count - target_count) / max(target_count, 1)
        return max(0, 1.0 - word_diff * 0.5)
    
    def _calculate_usage_score(self, has_feature: bool, pattern_percentage: float, threshold: float) -> float:
        """Calculate score for binary features like questions, numbers, etc."""
        if pattern_percentage > threshold:
            return 1.0 if has_feature else 0.3
        else:
            return 0.3 if has_feature else 1.0
    
    def _calculate_capitalization_score(self, title: str, patterns: IntelligentPatterns) -> float:
        """Calculate capitalization alignment score"""
        words = title.split()
        if not words:
            return 0.5
        
        capitalized_ratio = sum(1 for word in words if word[0].isupper()) / len(words)
        target_ratio = patterns.capitalization_score
        cap_diff = abs(capitalized_ratio - target_ratio)
        return max(0, 1.0 - cap_diff)
    
    def _calculate_keyword_score(self, title: str, patterns: IntelligentPatterns) -> float:
        """Calculate keyword alignment score"""
        title_words = set(re.findall(r'\b\w+\b', title.lower()))
        top_keywords = set(patterns.top_keywords)
        
        if not top_keywords:
            return 0.5
        
        keyword_matches = len(title_words.intersection(top_keywords))
        return min(1.0, keyword_matches / max(1, len(top_keywords) * 0.3))
    
    def _calculate_overall_score(self, pattern_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        total_weight = 0.0
        
        for pattern, score in pattern_scores.items():
            weight = self.score_weights.get(pattern, 0.0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5  # Default score if no weights
    
    def _calculate_title_confidence(self, title: str, patterns: IntelligentPatterns, 
                                  pattern_scores: Dict[str, float]) -> float:
        """Calculate confidence in the title evaluation"""
        
        # Base confidence from pattern analysis
        base_confidence = patterns.confidence_score
        
        # Confidence boost from pattern alignment
        avg_pattern_score = np.mean(list(pattern_scores.values()))
        alignment_confidence = avg_pattern_score * 0.3
        
        # Confidence from title characteristics
        title_length = len(title)
        if 20 <= title_length <= 100:  # Reasonable length
            length_confidence = 0.1
        else:
            length_confidence = 0.0
        
        # Combine confidences
        total_confidence = min(1.0, base_confidence + alignment_confidence + length_confidence)
        
        return total_confidence
    
    def _predict_performance_category(self, overall_score: float, confidence: float) -> str:
        """Predict performance category based on score and confidence"""
        
        # Adjust thresholds based on confidence
        high_threshold = 0.75 - (0.1 * (1.0 - confidence))
        medium_threshold = 0.55 - (0.1 * (1.0 - confidence))
        
        if overall_score >= high_threshold:
            return "high"
        elif overall_score >= medium_threshold:
            return "medium"
        else:
            return "low"
    
    def _identify_strengths(self, title: str, pattern_scores: Dict[str, float], 
                          patterns: IntelligentPatterns) -> List[str]:
        """Identify the strengths of the title"""
        strengths = []
        
        # Check each pattern score
        if pattern_scores.get('word_count', 0) > 0.8:
            strengths.append(f"Optimal length ({len(title.split())} words matches channel pattern)")
        
        if pattern_scores.get('question_usage', 0) > 0.8:
            if '?' in title:
                strengths.append("Effective use of question format")
            else:
                strengths.append("Correctly avoids question format (not effective for this channel)")
        
        if pattern_scores.get('numeric_usage', 0) > 0.8:
            if re.search(r'\d+', title):
                strengths.append("Strategic use of numbers")
            else:
                strengths.append("Correctly avoids numbers (not effective for this channel)")
        
        if pattern_scores.get('keyword_match', 0) > 0.7:
            strengths.append("Strong keyword alignment with successful titles")
        
        if pattern_scores.get('capitalization', 0) > 0.8:
            strengths.append("Optimal capitalization pattern")
        
        # Overall strengths
        if len([s for s in pattern_scores.values() if s > 0.7]) >= 3:
            strengths.append("Strong alignment with multiple success patterns")
        
        return strengths
    
    def _generate_recommendations(self, title: str, pattern_scores: Dict[str, float], 
                                patterns: IntelligentPatterns) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        # Add specific recommendations for each low-scoring pattern
        recommendations.extend(self._get_word_count_recommendations(title, pattern_scores, patterns))
        recommendations.extend(self._get_usage_recommendations(title, pattern_scores, patterns))
        recommendations.extend(self._get_keyword_recommendations(pattern_scores, patterns))
        
        # Add general recommendation if overall score is low
        overall_score = np.mean(list(pattern_scores.values()))
        if overall_score < 0.6:
            recommendations.append("Focus on the highest-weighted patterns for this channel to improve performance")
        
        return recommendations
    
    def _get_word_count_recommendations(self, title: str, pattern_scores: Dict[str, float], 
                                      patterns: IntelligentPatterns) -> List[str]:
        """Get word count specific recommendations"""
        recommendations = []
        if pattern_scores.get('word_count', 0) < 0.6:
            current_count = len(title.split())
            target_count = int(patterns.avg_word_count)
            if current_count < target_count - 2:
                recommendations.append(f"Consider adding 1-2 more words (current: {current_count}, target: ~{target_count})")
            elif current_count > target_count + 2:
                recommendations.append(f"Consider shortening by 1-2 words (current: {current_count}, target: ~{target_count})")
        return recommendations
    
    def _get_usage_recommendations(self, title: str, pattern_scores: Dict[str, float], 
                                 patterns: IntelligentPatterns) -> List[str]:
        """Get recommendations for questions, numbers, etc."""
        recommendations = []
        
        if pattern_scores.get('question_usage', 0) < 0.6:
            if patterns.question_percentage > 0.1 and '?' not in title:
                recommendations.append("Consider using a question format - questions perform well for this channel")
            elif patterns.question_percentage <= 0.1 and '?' in title:
                recommendations.append("Consider removing question format - statements perform better for this channel")
        
        if pattern_scores.get('numeric_usage', 0) < 0.6:
            if patterns.numeric_percentage > 0.2 and not re.search(r'\d+', title):
                recommendations.append("Consider adding specific numbers - they're effective for this channel")
        
        return recommendations
    
    def _get_keyword_recommendations(self, pattern_scores: Dict[str, float], 
                                   patterns: IntelligentPatterns) -> List[str]:
        """Get keyword-specific recommendations"""
        recommendations = []
        if pattern_scores.get('keyword_match', 0) < 0.5:
            top_keywords = patterns.top_keywords[:3]
            recommendations.append(f"Consider incorporating high-performing keywords: {', '.join(top_keywords)}")
        return recommendations
    
    def _generate_detailed_reasoning(self, title: str, pattern_scores: Dict[str, float], 
                                   patterns: IntelligentPatterns) -> str:
        """Generate detailed reasoning for the title evaluation"""
        
        reasoning_parts = []
        
        # Overall assessment
        overall_score = np.mean(list(pattern_scores.values()))
        if overall_score > 0.75:
            reasoning_parts.append("Strong alignment with successful patterns from this channel.")
        elif overall_score > 0.6:
            reasoning_parts.append("Good alignment with channel patterns, with room for optimization.")
        else:
            reasoning_parts.append("Moderate alignment with patterns - consider adjustments.")
        
        # Specific pattern analysis
        strong_patterns = [k for k, v in pattern_scores.items() if v > 0.7]
        weak_patterns = [k for k, v in pattern_scores.items() if v < 0.5]
        
        if strong_patterns:
            reasoning_parts.append(f"Strong in: {', '.join(strong_patterns)}.")
        
        if weak_patterns:
            reasoning_parts.append(f"Could improve: {', '.join(weak_patterns)}.")
        
        # Confidence assessment
        if patterns.confidence_score > 0.7:
            reasoning_parts.append("High confidence in predictions due to robust channel data.")
        elif patterns.confidence_score < 0.4:
            reasoning_parts.append("Lower confidence due to limited channel data - general best practices applied.")
        
        return " ".join(reasoning_parts)