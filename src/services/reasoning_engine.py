"""
Enhanced Reasoning Engine for TitleCraft AI

This module provides detailed explanations and reasoning for title generation decisions,
including specific citations from successful examples and pattern analysis.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re

from ..data.models import ChannelProfile, VideoData
from .quality_scorer import QualityScore
from ..processing.semantic_matcher import SimilarityMatch

logger = logging.getLogger(__name__)


@dataclass
class ReasoningCitation:
    """Citation from a successful example"""
    source_title: str
    source_views: int
    pattern_matched: str
    relevance_score: float
    explanation: str


@dataclass
class PatternEvidence:
    """Evidence supporting a pattern choice"""
    pattern_name: str
    usage_frequency: float
    performance_impact: float
    examples: List[str]
    confidence: float
    recommendation: str


@dataclass
class TitleReasoning:
    """Comprehensive reasoning for a generated title"""
    title: str
    overall_confidence: float
    strategy_used: str
    key_decisions: List[str]
    pattern_evidence: List[PatternEvidence]
    citations: List[ReasoningCitation]
    success_factors: List[str]
    potential_risks: List[str]
    improvement_opportunities: List[str]
    expected_performance: Dict[str, float]
    reasoning_summary: str


class EnhancedReasoningEngine:
    """
    Advanced reasoning engine that provides detailed explanations for title generation.
    
    Features:
    - Pattern-based reasoning with evidence
    - Citations from successful examples
    - Performance prediction explanations
    - Risk assessment reasoning
    - Strategic decision explanations
    """
    
    def __init__(self):
        self.reasoning_templates = self._initialize_reasoning_templates()
        self.pattern_descriptions = self._initialize_pattern_descriptions()
    
    def _initialize_reasoning_templates(self) -> Dict[str, str]:
        """Initialize templates for different types of reasoning"""
        return {
            'pattern_match': "This title uses the '{pattern}' pattern, which appears in {frequency:.0%} of your successful videos and performs {performance:.1f}x better than average.",
            
            'emotional_trigger': "The emotional trigger '{trigger}' was chosen because it appears in {examples_count} of your top-performing videos, including '{example_title}' ({views:,} views).",
            
            'length_optimization': "The {word_count}-word length aligns with your channel's optimal range of {optimal_range} words, based on analysis of {video_count} videos.",
            
            'search_optimization': "Keywords '{keywords}' were included to match search intent, similar to your successful video '{example}' which likely benefits from search traffic.",
            
            'audience_alignment': "The {complexity_level} complexity level matches your audience's preferences, evidenced by consistent performance of similar titles.",
            
            'competitive_advantage': "This approach differentiates from common industry patterns while maintaining elements that work for your specific audience.",
            
            'risk_mitigation': "Certain high-risk elements were avoided based on platform guidelines and your channel's brand positioning."
        }
    
    def _initialize_pattern_descriptions(self) -> Dict[str, Dict[str, str]]:
        """Initialize detailed pattern descriptions and success factors"""
        return {
            'numbered_list': {
                'description': 'Titles starting with numbers (e.g., "5 Ways", "10 Things")',
                'success_factors': ['Clear expectation setting', 'Easy to scan', 'Promise of organized content'],
                'optimal_range': '3-20',
                'psychological_appeal': 'Numbers create concrete expectations and suggest comprehensive coverage'
            },
            
            'how_to': {
                'description': 'Instructional format titles (e.g., "How to", "How I")',
                'success_factors': ['Clear value proposition', 'Search-friendly', 'Educational appeal'],
                'optimal_context': 'Tutorial and educational content',
                'psychological_appeal': 'Promises practical, actionable information'
            },
            
            'question_format': {
                'description': 'Question-style titles that create curiosity',
                'success_factors': ['Curiosity generation', 'Conversational tone', 'Search alignment'],
                'optimal_context': 'Exploratory and analytical content',
                'psychological_appeal': 'Engages natural human curiosity and need for answers'
            },
            
            'comparison': {
                'description': 'Comparative titles (e.g., "A vs B", "Before vs After")',
                'success_factors': ['Clear differentiation', 'Decision-helping', 'Comprehensive coverage'],
                'optimal_context': 'Review and analysis content',
                'psychological_appeal': 'Helps viewers make informed decisions'
            },
            
            'emotional_intensity': {
                'description': 'High-emotion words that amplify impact',
                'success_factors': ['Increased click-through', 'Memorable impact', 'Shareability'],
                'risk_factors': ['Potential clickbait perception', 'Must deliver on promise'],
                'psychological_appeal': 'Triggers emotional response and urgency'
            }
        }
    
    def generate_reasoning(self,
                          title: str,
                          channel_profile: ChannelProfile,
                          similar_examples: List[VideoData],
                          enhanced_profile: Dict[str, Any],
                          quality_score: QualityScore,
                          similarity_matches: List[SimilarityMatch] = None) -> TitleReasoning:
        """
        Generate comprehensive reasoning for a title choice
        
        Args:
            title: The generated title to explain
            channel_profile: Channel profile data
            similar_examples: Similar successful videos
            enhanced_profile: Enhanced pattern analysis
            quality_score: Quality assessment of the title
            similarity_matches: Semantic similarity matches
            
        Returns:
            Detailed reasoning explanation
        """
        
        # Analyze title patterns and decisions
        key_decisions = self._identify_key_decisions(title, enhanced_profile)
        
        # Generate pattern evidence
        pattern_evidence = self._generate_pattern_evidence(
            title, channel_profile, enhanced_profile
        )
        
        # Create citations from examples
        citations = self._create_citations(
            title, similar_examples, similarity_matches, enhanced_profile
        )
        
        # Identify success factors
        success_factors = self._identify_success_factors(
            title, channel_profile, enhanced_profile, quality_score
        )
        
        # Assess risks
        potential_risks = self._assess_potential_risks(title, quality_score)
        
        # Find improvement opportunities
        improvement_opportunities = self._identify_improvements(
            title, quality_score, enhanced_profile
        )
        
        # Calculate expected performance
        expected_performance = self._calculate_expected_performance(
            title, channel_profile, quality_score, pattern_evidence
        )
        
        # Determine strategy used
        strategy_used = self._determine_strategy(title, pattern_evidence)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_reasoning_confidence(
            pattern_evidence, citations, quality_score
        )
        
        # Generate comprehensive summary
        reasoning_summary = self._generate_reasoning_summary(
            title, strategy_used, pattern_evidence, citations, 
            success_factors, expected_performance
        )
        
        return TitleReasoning(
            title=title,
            overall_confidence=overall_confidence,
            strategy_used=strategy_used,
            key_decisions=key_decisions,
            pattern_evidence=pattern_evidence,
            citations=citations,
            success_factors=success_factors,
            potential_risks=potential_risks,
            improvement_opportunities=improvement_opportunities,
            expected_performance=expected_performance,
            reasoning_summary=reasoning_summary
        )
    
    def _identify_key_decisions(self, 
                               title: str, 
                               enhanced_profile: Dict[str, Any]) -> List[str]:
        """Identify key strategic decisions made in title generation"""
        
        decisions = []
        title_lower = title.lower()
        
        # Length decision
        word_count = len(title.split())
        if word_count <= 5:
            decisions.append(f"Chose concise {word_count}-word format for immediate impact")
        elif word_count >= 10:
            decisions.append(f"Used detailed {word_count}-word format for comprehensive description")
        else:
            decisions.append(f"Selected balanced {word_count}-word length for optimal engagement")
        
        # Pattern decisions
        if re.search(r'^\\d+', title):
            number = re.search(r'^(\\d+)', title).group(1)
            decisions.append(f"Started with number '{number}' to create clear expectations")
        
        if title.strip().endswith('?'):
            decisions.append("Used question format to engage curiosity and encourage interaction")
        
        if 'how to' in title_lower:
            decisions.append("Applied instructional format to promise actionable value")
        
        if any(word in title_lower for word in ['vs', 'versus', 'against']):
            decisions.append("Implemented comparison format to help viewers make decisions")
        
        # Emotional decisions
        emotional_words = ['amazing', 'incredible', 'shocking', 'secret', 'hidden', 'ultimate']
        found_emotions = [word for word in emotional_words if word in title_lower]
        if found_emotions:
            decisions.append(f"Included emotional triggers ({', '.join(found_emotions)}) to increase click appeal")
        
        # SEO decisions
        if any(phrase in title_lower for phrase in ['best', 'top', 'review', 'guide']):
            decisions.append("Incorporated search-friendly keywords for discoverability")
        
        return decisions
    
    def _generate_pattern_evidence(self,
                                  title: str,
                                  channel_profile: ChannelProfile,
                                  enhanced_profile: Dict[str, Any]) -> List[PatternEvidence]:
        """Generate evidence supporting pattern choices"""
        
        evidence = []
        
        if enhanced_profile and 'structural_patterns' in enhanced_profile:
            patterns = enhanced_profile['structural_patterns']
            
            for pattern_name, pattern_data in patterns.items():
                if self._title_matches_pattern(title, pattern_name):
                    
                    usage_freq = pattern_data.get('usage_rate', 0)
                    performance_impact = pattern_data.get('performance_vs_average', 1)
                    examples = pattern_data.get('examples', [])
                    
                    # Calculate confidence based on data quality
                    confidence = min(usage_freq * 2 + (performance_impact - 1), 1.0)
                    
                    # Generate recommendation
                    if performance_impact > 1.2:
                        recommendation = "Highly recommended - strong performance indicator"
                    elif performance_impact > 1.0:
                        recommendation = "Recommended - positive performance correlation"
                    else:
                        recommendation = "Use with caution - lower performance correlation"
                    
                    evidence.append(PatternEvidence(
                        pattern_name=pattern_name,
                        usage_frequency=usage_freq,
                        performance_impact=performance_impact,
                        examples=examples[:3],  # Top 3 examples
                        confidence=confidence,
                        recommendation=recommendation
                    ))
        
        return evidence
    
    def _title_matches_pattern(self, title: str, pattern_name: str) -> bool:
        """Check if title matches a specific pattern"""
        title_lower = title.lower()
        
        pattern_checks = {
            'numbered_list': lambda t: bool(re.search(r'^\\d+', t)),
            'how_to': lambda t: 'how to' in t or 'how i' in t,
            'question_format': lambda t: t.strip().endswith('?'),
            'comparison': lambda t: any(word in t for word in ['vs', 'versus', 'against']),
            'superlative': lambda t: any(word in t for word in ['best', 'worst', 'top', 'ultimate']),
            'emotional_intensity': lambda t: any(word in t for word in ['shocking', 'amazing', 'incredible'])
        }
        
        checker = pattern_checks.get(pattern_name)
        return checker(title_lower) if checker else False
    
    def _create_citations(self,
                         title: str,
                         similar_examples: List[VideoData],
                         similarity_matches: List[SimilarityMatch],
                         enhanced_profile: Dict[str, Any]) -> List[ReasoningCitation]:
        """Create citations from successful examples"""
        
        citations = []
        
        # Use similarity matches if available
        if similarity_matches:
            for match in similarity_matches[:3]:  # Top 3 matches
                pattern_matched = self._identify_matched_pattern(title, match.video.title)
                
                citation = ReasoningCitation(
                    source_title=match.video.title,
                    source_views=match.video.views_in_period,
                    pattern_matched=pattern_matched,
                    relevance_score=match.similarity_score,
                    explanation=f"Similar {pattern_matched} pattern with {match.similarity_score:.1%} relevance"
                )
                citations.append(citation)
        
        # Fallback to similar examples
        elif similar_examples:
            for example in similar_examples[:3]:
                pattern_matched = self._identify_matched_pattern(title, example.title)
                
                citation = ReasoningCitation(
                    source_title=example.title,
                    source_views=example.views_in_period,
                    pattern_matched=pattern_matched,
                    relevance_score=0.7,  # Default relevance
                    explanation=f"Successful example using {pattern_matched} pattern"
                )
                citations.append(citation)
        
        return citations
    
    def _identify_matched_pattern(self, title1: str, title2: str) -> str:
        """Identify what pattern two titles have in common"""
        
        title1_lower = title1.lower()
        title2_lower = title2.lower()
        
        # Check for common patterns
        if re.search(r'^\\d+', title1) and re.search(r'^\\d+', title2):
            return "numbered list format"
        
        if 'how to' in title1_lower and 'how to' in title2_lower:
            return "instructional format"
        
        if title1.strip().endswith('?') and title2.strip().endswith('?'):
            return "question format"
        
        if any(word in title1_lower and word in title2_lower for word in ['vs', 'versus']):
            return "comparison format"
        
        # Check for emotional words
        emotional_words = ['amazing', 'incredible', 'shocking', 'secret', 'best', 'ultimate']
        common_emotions = [word for word in emotional_words 
                          if word in title1_lower and word in title2_lower]
        if common_emotions:
            return f"emotional trigger ({common_emotions[0]})"
        
        # Check for word overlap
        words1 = set(title1_lower.split())
        words2 = set(title2_lower.split())
        common_words = words1.intersection(words2)
        
        if len(common_words) >= 2:
            return f"keyword similarity ({', '.join(list(common_words)[:2])})"
        
        return "thematic similarity"
    
    def _identify_success_factors(self,
                                 title: str,
                                 channel_profile: ChannelProfile,
                                 enhanced_profile: Dict[str, Any],
                                 quality_score: QualityScore) -> List[str]:
        """Identify factors that contribute to title's success potential"""
        
        factors = []
        
        # Quality score based factors
        for dimension, score in quality_score.dimension_scores.items():
            if score >= 0.8:
                factor_descriptions = {
                    'clickability': 'Strong emotional appeal and curiosity generation',
                    'searchability': 'Excellent SEO optimization and keyword usage',
                    'channel_fit': 'Perfect alignment with channel patterns and audience',
                    'readability': 'Clear, accessible language that\'s easy to understand',
                    'uniqueness': 'Original approach that stands out from competitors',
                    'risk_assessment': 'Brand-safe content with minimal policy risks'
                }
                factors.append(factor_descriptions.get(dimension, f'Strong {dimension}'))
        
        # Pattern-specific success factors
        title_lower = title.lower()
        
        if re.search(r'^\\d+', title):
            factors.append('Numbered format creates clear expectations and improves scannability')
        
        if title.strip().endswith('?'):
            factors.append('Question format engages natural curiosity and encourages interaction')
        
        if 'how to' in title_lower:
            factors.append('Instructional format promises actionable, practical value')
        
        # Length optimization
        word_count = len(title.split())
        optimal_length = channel_profile.title_patterns.avg_length_words
        if abs(word_count - optimal_length) <= 2:
            factors.append('Length optimized for this channel\'s audience preferences')
        
        # Emotional triggers
        emotional_words = ['secret', 'amazing', 'shocking', 'incredible', 'ultimate', 'best']
        found_emotions = [word for word in emotional_words if word in title_lower]
        if found_emotions:
            factors.append(f'Emotional triggers ({", ".join(found_emotions)}) increase engagement potential')
        
        return factors
    
    def _assess_potential_risks(self, title: str, quality_score: QualityScore) -> List[str]:
        """Assess potential risks and concerns with the title"""
        
        risks = []
        
        # Use risk assessment from quality score
        if quality_score.risk_factors:
            risks.extend(quality_score.risk_factors)
        
        # Additional risk analysis
        title_lower = title.lower()
        
        # Clickbait risks
        clickbait_indicators = ['you won\'t believe', 'shocking', 'doctors hate', 'this will']
        found_clickbait = [indicator for indicator in clickbait_indicators if indicator in title_lower]
        if found_clickbait:
            risks.append(f'Potential clickbait perception from phrases: {", ".join(found_clickbait)}')
        
        # Over-promising risks
        extreme_words = ['always', 'never', 'guaranteed', 'instant', 'miracle']
        found_extreme = [word for word in extreme_words if word in title_lower]
        if found_extreme:
            risks.append(f'May over-promise with words: {", ".join(found_extreme)}')
        
        # Length risks
        if len(title) > 100:
            risks.append('Title may be truncated in search results and recommendations')
        
        if len(title.split()) > 15:
            risks.append('Long title may reduce readability and impact')
        
        # Complexity risks
        complex_words = [word for word in title.split() if len(word) > 10]
        if len(complex_words) > 2:
            risks.append('Complex vocabulary may limit accessibility')
        
        return risks
    
    def _identify_improvements(self,
                              title: str,
                              quality_score: QualityScore,
                              enhanced_profile: Dict[str, Any]) -> List[str]:
        """Identify opportunities for title improvement"""
        
        improvements = []
        
        # Use improvement suggestions from quality score
        if quality_score.improvement_suggestions:
            improvements.extend(quality_score.improvement_suggestions)
        
        # Additional improvement analysis
        title_lower = title.lower()
        
        # Emotional enhancement
        emotional_words = ['secret', 'amazing', 'shocking', 'incredible']
        if not any(word in title_lower for word in emotional_words):
            improvements.append('Consider adding emotional trigger words to increase click appeal')
        
        # Number optimization
        if not re.search(r'\\d+', title):
            improvements.append('Adding specific numbers could improve concrete appeal')
        
        # Question format opportunity
        if not title.strip().endswith('?') and 'how' not in title_lower:
            improvements.append('Question format could increase curiosity and engagement')
        
        # Search optimization
        search_terms = ['best', 'review', 'guide', 'tutorial', 'tips']
        if not any(term in title_lower for term in search_terms):
            improvements.append('Including search-friendly terms could improve discoverability')
        
        return improvements
    
    def _calculate_expected_performance(self,
                                       title: str,
                                       channel_profile: ChannelProfile,
                                       quality_score: QualityScore,
                                       pattern_evidence: List[PatternEvidence]) -> Dict[str, float]:
        """Calculate expected performance metrics"""
        
        base_views = channel_profile.stats.avg_views
        
        # Performance multiplier based on quality score
        quality_multiplier = 0.5 + (quality_score.overall_score * 1.5)
        
        # Pattern-based multiplier
        pattern_multiplier = 1.0
        for evidence in pattern_evidence:
            if evidence.confidence > 0.7:
                pattern_multiplier *= evidence.performance_impact
        
        pattern_multiplier = min(pattern_multiplier, 2.0)  # Cap at 2x
        
        # Calculate expected metrics
        expected_views = base_views * quality_multiplier * pattern_multiplier
        expected_ctr = 3.0 + (quality_score.dimension_scores.get('clickability', 0.5) * 5.0)
        expected_engagement = 0.02 + (quality_score.overall_score * 0.03)
        
        return {
            'views': expected_views,
            'click_through_rate': expected_ctr,
            'engagement_rate': expected_engagement,
            'performance_vs_average': expected_views / base_views if base_views > 0 else 1.0
        }
    
    def _determine_strategy(self, title: str, pattern_evidence: List[PatternEvidence]) -> str:
        """Determine the primary strategy used in title generation"""
        
        title_lower = title.lower()
        
        # Identify primary strategy based on patterns and content
        if any(evidence.pattern_name == 'numbered_list' for evidence in pattern_evidence):
            return "List-Based Engagement"
        
        if 'how to' in title_lower:
            return "Educational Value"
        
        if title.strip().endswith('?'):
            return "Curiosity-Driven"
        
        if any(word in title_lower for word in ['vs', 'versus', 'against']):
            return "Comparative Analysis"
        
        if any(word in title_lower for word in ['best', 'top', 'ultimate']):
            return "Authority Positioning"
        
        if any(word in title_lower for word in ['secret', 'hidden', 'shocking']):
            return "Exclusive Reveal"
        
        return "Balanced Optimization"
    
    def _calculate_reasoning_confidence(self,
                                      pattern_evidence: List[PatternEvidence],
                                      citations: List[ReasoningCitation],
                                      quality_score: QualityScore) -> float:
        """Calculate overall confidence in the reasoning"""
        
        # Evidence quality
        evidence_confidence = 0.0
        if pattern_evidence:
            evidence_confidence = sum(e.confidence for e in pattern_evidence) / len(pattern_evidence)
        
        # Citation quality
        citation_confidence = 0.0
        if citations:
            citation_confidence = sum(c.relevance_score for c in citations) / len(citations)
        
        # Quality score confidence
        score_confidence = quality_score.confidence_level
        
        # Weighted average
        overall_confidence = (
            evidence_confidence * 0.4 +
            citation_confidence * 0.3 +
            score_confidence * 0.3
        )
        
        return min(overall_confidence, 1.0)
    
    def _generate_reasoning_summary(self,
                                   title: str,
                                   strategy: str,
                                   pattern_evidence: List[PatternEvidence],
                                   citations: List[ReasoningCitation],
                                   success_factors: List[str],
                                   expected_performance: Dict[str, float]) -> str:
        """Generate a comprehensive reasoning summary"""
        
        performance_ratio = expected_performance.get('performance_vs_average', 1.0)
        
        # Opening statement
        summary = f'**Strategy: {strategy}**\\n\\n'
        summary += f'This title is designed to achieve {performance_ratio:.1f}x your channel average '
        summary += f'({expected_performance.get("views", 0):,.0f} estimated views).\\n\\n'
        
        # Key evidence
        if pattern_evidence:
            summary += '**Pattern Evidence:**\\n'
            for evidence in pattern_evidence[:2]:  # Top 2 pieces of evidence
                summary += f'• {evidence.pattern_name}: {evidence.performance_impact:.1f}x performance, '
                summary += f'{evidence.usage_frequency:.0%} channel usage ({evidence.recommendation})\\n'
        
        # Citations
        if citations:
            summary += '\\n**Successful Examples:**\\n'
            for citation in citations[:2]:  # Top 2 citations
                summary += f'• "{citation.source_title}" ({citation.source_views:,} views) - '
                summary += f'{citation.explanation}\\n'
        
        # Success factors
        if success_factors:
            summary += '\\n**Key Success Factors:**\\n'
            for factor in success_factors[:3]:  # Top 3 factors
                summary += f'• {factor}\\n'
        
        # Performance prediction
        summary += f'\\n**Expected Performance:**\\n'
        summary += f'• Click-through rate: {expected_performance.get("click_through_rate", 0):.1f}%\\n'
        summary += f'• Engagement rate: {expected_performance.get("engagement_rate", 0):.1%}\\n'
        summary += f'• Performance vs average: {performance_ratio:.1f}x'
        
        return summary
    
    def compare_title_reasoning(self,
                               titles_with_reasoning: List[Tuple[str, TitleReasoning]]) -> Dict[str, Any]:
        """Compare reasoning across multiple titles"""
        
        comparison = {
            'best_overall': None,
            'highest_confidence': None,
            'most_evidence': None,
            'best_performance': None,
            'strategy_distribution': {},
            'common_success_factors': [],
            'summary': ""
        }
        
        if not titles_with_reasoning:
            return comparison
        
        # Find best performers in each category
        best_overall = max(titles_with_reasoning, 
                          key=lambda x: x[1].overall_confidence)
        comparison['best_overall'] = best_overall[0]
        
        highest_confidence = max(titles_with_reasoning,
                               key=lambda x: x[1].overall_confidence)
        comparison['highest_confidence'] = highest_confidence[0]
        
        most_evidence = max(titles_with_reasoning,
                          key=lambda x: len(x[1].pattern_evidence))
        comparison['most_evidence'] = most_evidence[0]
        
        best_performance = max(titles_with_reasoning,
                             key=lambda x: x[1].expected_performance.get('performance_vs_average', 1.0))
        comparison['best_performance'] = best_performance[0]
        
        # Strategy distribution
        strategies = [reasoning.strategy_used for _, reasoning in titles_with_reasoning]
        comparison['strategy_distribution'] = {
            strategy: strategies.count(strategy) for strategy in set(strategies)
        }
        
        # Common success factors
        all_factors = []
        for _, reasoning in titles_with_reasoning:
            all_factors.extend(reasoning.success_factors)
        
        factor_counts = {}
        for factor in all_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        common_factors = [factor for factor, count in factor_counts.items() 
                         if count >= len(titles_with_reasoning) * 0.5]
        comparison['common_success_factors'] = common_factors
        
        # Generate summary
        comparison['summary'] = self._generate_comparison_summary(
            titles_with_reasoning, comparison
        )
        
        return comparison
    
    def _generate_comparison_summary(self,
                                    titles_with_reasoning: List[Tuple[str, TitleReasoning]],
                                    comparison: Dict[str, Any]) -> str:
        """Generate a summary comparing multiple title reasoning"""
        
        summary = f"**Analysis of {len(titles_with_reasoning)} Generated Titles**\\n\\n"
        
        # Best performer
        best_title = comparison['best_overall']
        best_reasoning = next(r for t, r in titles_with_reasoning if t == best_title)
        expected_perf = best_reasoning.expected_performance.get('performance_vs_average', 1.0)
        
        summary += f"**Recommended Title:** '{best_title}'\\n"
        summary += f"Expected to perform {expected_perf:.1f}x channel average\\n\\n"
        
        # Strategy distribution
        summary += "**Strategy Distribution:**\\n"
        for strategy, count in comparison['strategy_distribution'].items():
            summary += f"• {strategy}: {count} titles\\n"
        
        # Common success factors
        if comparison['common_success_factors']:
            summary += "\\n**Common Success Factors:**\\n"
            for factor in comparison['common_success_factors'][:3]:
                summary += f"• {factor}\\n"
        
        return summary