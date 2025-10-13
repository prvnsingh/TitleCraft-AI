"""
Title Generation Engine for TitleCraft AI
Main orchestrator that coordinates pattern analysis, LLM generation, and quality scoring
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import time
from datetime import datetime

from ..utils import BaseComponent
from ..data import DataStore, ChannelProfileManager, VideoData, ChannelProfile
from ..processing.pattern_profiler import EnhancedPatternProfiler
from ..processing.llm_orchestrator import (
    LLMOrchestrator, 
    TitleGenerationRequest, 
    TitleGenerationResponse,
    GeneratedTitle
)

@dataclass
class TitleRequest:
    """Main request structure for title generation"""
    idea: str
    channel_id: str
    n_titles: int = 4
    context: Optional[str] = None
    style_preferences: Optional[Dict[str, Any]] = None
    force_profile_update: bool = False

@dataclass
class TitleResult:
    """Individual title result with enhanced metadata"""
    title: str
    reasoning: str
    confidence: float
    quality_score: float
    pattern_analysis: Dict[str, Any]
    performance_prediction: Optional[float] = None
    
class TitleGenerationEngine(BaseComponent):
    """
    Main title generation engine that orchestrates the entire pipeline.
    
    Pipeline flow:
    1. Load/create channel profile
    2. Find contextually similar examples
    3. Generate titles using LLM
    4. Score and rank titles
    5. Provide detailed reasoning
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Initialize core components
        self.data_store = DataStore(config=self.config)
        self.profile_manager = ChannelProfileManager(self.data_store)
        self.pattern_profiler = EnhancedPatternProfiler(config=self.config)
        self.llm_orchestrator = LLMOrchestrator(config=self.config)
        
        # Performance tracking
        self.generation_metrics = {
            'total_generations': 0,
            'avg_generation_time': 0.0,
            'successful_generations': 0,
            'failed_generations': 0
        }
        
        self.logger.info("TitleGenerationEngine initialized")
    
    async def generate_titles(self, request: TitleRequest) -> Dict[str, Any]:
        """
        Generate titles for a given video idea and channel.
        
        Args:
            request: TitleRequest with idea, channel_id, and preferences
            
        Returns:
            Dictionary with generated titles, analysis, and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Generating titles for channel {request.channel_id}: '{request.idea}'")
            
            # Step 1: Get or create channel profile
            channel_profile = await self._get_channel_profile(
                request.channel_id, 
                force_update=request.force_profile_update
            )
            
            # Step 2: Find similar examples
            similar_examples = await self._find_similar_examples(
                request.idea, 
                request.channel_id
            )
            
            # Step 3: Generate titles using LLM
            llm_request = TitleGenerationRequest(
                idea=request.idea,
                channel_id=request.channel_id,
                n_titles=request.n_titles,
                style_preferences=request.style_preferences,
                context=request.context
            )
            
            llm_response = await self.llm_orchestrator.generate_titles(
                llm_request, channel_profile, similar_examples
            )
            
            # Step 4: Enhance titles with quality scoring and analysis
            enhanced_titles = await self._enhance_generated_titles(
                llm_response.titles,
                channel_profile,
                similar_examples,
                request.idea
            )
            
            # Step 5: Rank titles by overall score
            ranked_titles = self._rank_titles(enhanced_titles)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update metrics
            self._update_generation_metrics(processing_time, success=True)
            
            # Create comprehensive response
            response = {
                'titles': [asdict(title) for title in ranked_titles],
                'channel_analysis': self._create_channel_analysis_summary(channel_profile),
                'similar_examples': [asdict(ex) for ex in similar_examples],
                'generation_insights': {
                    'processing_time': processing_time,
                    'model_used': llm_response.model_used,
                    'confidence_range': self._calculate_confidence_range(enhanced_titles),
                    'pattern_usage_summary': self._summarize_pattern_usage(enhanced_titles),
                    'recommendations': self._generate_usage_recommendations(
                        enhanced_titles, channel_profile
                    )
                },
                'metadata': {
                    'request_id': llm_response.request_id,
                    'generation_timestamp': datetime.now().isoformat(),
                    'channel_id': request.channel_id,
                    'original_idea': request.idea,
                    'titles_requested': request.n_titles,
                    'titles_generated': len(enhanced_titles)
                }
            }
            
            self.logger.info(f"Successfully generated {len(enhanced_titles)} titles in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_generation_metrics(processing_time, success=False)
            
            self.logger.error(f"Title generation failed: {e}")
            
            # Return error response with fallback
            return self._create_error_response(request, str(e), processing_time)
    
    async def _get_channel_profile(self, 
                                 channel_id: str, 
                                 force_update: bool = False) -> ChannelProfile:
        """Get or create channel profile"""
        
        try:
            if force_update:
                profile = self.profile_manager.create_channel_profile(
                    channel_id, force_update=True
                )
            else:
                # Try to get existing profile first
                profile = self.profile_manager.get_profile_by_channel(channel_id)
                
                if profile is None:
                    # Create new profile
                    profile = self.profile_manager.create_channel_profile(channel_id)
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to get channel profile for {channel_id}: {e}")
            # Return a basic fallback profile
            return self._create_fallback_profile(channel_id)
    
    async def _find_similar_examples(self, 
                                   idea: str, 
                                   channel_id: str) -> List[VideoData]:
        """Find contextually similar high-performing examples"""
        
        try:
            # Get channel data
            channel_data = self.data_store.get_channel_data(channel_id)
            
            # Get high performers
            high_performers = self.data_store.get_high_performers(
                channel_id=channel_id, 
                threshold_percentile=0.7,
                data=channel_data
            )
            
            # Simple content-based similarity (can be enhanced with embeddings)
            similar_videos = []
            idea_words = set(idea.lower().split())
            
            for _, video_row in high_performers.iterrows():
                title_words = set(video_row['title'].lower().split())
                summary_words = set(str(video_row.get('summary', '')).lower().split())
                
                # Calculate word overlap
                title_overlap = len(idea_words.intersection(title_words)) / len(idea_words) if idea_words else 0
                summary_overlap = len(idea_words.intersection(summary_words)) / len(idea_words) if idea_words else 0
                
                similarity_score = max(title_overlap, summary_overlap)
                
                if similarity_score > 0.1:  # Minimum similarity threshold
                    video_data = VideoData(
                        channel_id=video_row['channel_id'],
                        video_id=video_row['video_id'],
                        title=video_row['title'],
                        summary=str(video_row.get('summary', '')),
                        views_in_period=int(video_row['views_in_period'])
                    )
                    similar_videos.append((video_data, similarity_score))
            
            # Sort by similarity and return top examples
            similar_videos.sort(key=lambda x: x[1], reverse=True)
            
            # Return top N examples
            max_examples = self.config.data.max_similar_examples
            return [video for video, _ in similar_videos[:max_examples]]
            
        except Exception as e:
            self.logger.warning(f"Failed to find similar examples: {e}")
            return []
    
    async def _enhance_generated_titles(self,
                                      generated_titles: List[GeneratedTitle],
                                      channel_profile: ChannelProfile,
                                      similar_examples: List[VideoData],
                                      original_idea: str) -> List[TitleResult]:
        """Enhance generated titles with quality scoring and analysis"""
        
        enhanced_titles = []
        
        for title in generated_titles:
            try:
                # Analyze title patterns
                pattern_analysis = self._analyze_title_patterns(
                    title.title, channel_profile
                )
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(
                    title, channel_profile, pattern_analysis
                )
                
                # Predict performance (simplified)
                performance_prediction = self._predict_performance(
                    title, channel_profile, similar_examples
                )
                
                enhanced_title = TitleResult(
                    title=title.title,
                    reasoning=title.reasoning,
                    confidence=title.confidence,
                    quality_score=quality_score,
                    pattern_analysis=pattern_analysis,
                    performance_prediction=performance_prediction
                )
                
                enhanced_titles.append(enhanced_title)
                
            except Exception as e:
                self.logger.warning(f"Failed to enhance title '{title.title}': {e}")
                # Add with basic enhancement
                enhanced_titles.append(TitleResult(
                    title=title.title,
                    reasoning=title.reasoning,
                    confidence=title.confidence,
                    quality_score=0.5,
                    pattern_analysis={'error': str(e)},
                    performance_prediction=None
                ))
        
        return enhanced_titles
    
    def _analyze_title_patterns(self, 
                              title: str, 
                              channel_profile: ChannelProfile) -> Dict[str, Any]:
        """Analyze patterns in a specific title"""
        
        analysis = {
            'word_count': len(title.split()),
            'character_count': len(title),
            'question_format': '?' in title or title.lower().startswith(('why ', 'what ', 'how ', 'when ', 'where ')),
            'contains_numbers': bool(re.search(r'\d', title)),
            'superlatives': self._count_superlatives(title),
            'emotional_triggers': self._identify_emotional_triggers(title),
            'length_vs_optimal': self._compare_length_to_optimal(title, channel_profile),
            'pattern_adherence_score': 0.0
        }
        
        # Calculate pattern adherence score
        analysis['pattern_adherence_score'] = self._calculate_pattern_adherence(
            analysis, channel_profile
        )
        
        return analysis
    
    def _calculate_quality_score(self,
                               title: GeneratedTitle,
                               channel_profile: ChannelProfile,
                               pattern_analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score for a title"""
        
        scores = []
        
        # LLM confidence (30%)
        scores.append(title.confidence * 0.3)
        
        # Pattern adherence (40%)
        pattern_score = pattern_analysis.get('pattern_adherence_score', 0.5)
        scores.append(pattern_score * 0.4)
        
        # Length optimization (20%)
        length_score = self._score_title_length(title.title, channel_profile)
        scores.append(length_score * 0.2)
        
        # Uniqueness/creativity (10%)
        uniqueness_score = self._score_uniqueness(title.title)
        scores.append(uniqueness_score * 0.1)
        
        return sum(scores)
    
    def _predict_performance(self,
                           title: GeneratedTitle,
                           channel_profile: ChannelProfile,
                           similar_examples: List[VideoData]) -> Optional[float]:
        """Predict performance based on channel patterns"""
        
        if not similar_examples:
            return None
        
        try:
            # Simple performance prediction based on similar examples
            similar_views = [ex.views_in_period for ex in similar_examples]
            avg_similar_views = sum(similar_views) / len(similar_views)
            
            # Apply pattern-based modifiers
            pattern_modifier = 1.0
            
            # Length modifier
            optimal_length = channel_profile.title_patterns.avg_length_words
            title_length = len(title.title.split())
            length_diff = abs(title_length - optimal_length)
            
            if length_diff <= 1:
                pattern_modifier *= 1.1  # Close to optimal length
            elif length_diff > 3:
                pattern_modifier *= 0.9  # Far from optimal length
            
            # Pattern usage modifiers
            if title.pattern_matches:
                pattern_modifier *= 1.05  # Uses known patterns
            
            if title.emotional_triggers:
                pattern_modifier *= 1.02  # Uses emotional triggers
            
            predicted_views = avg_similar_views * pattern_modifier
            return float(predicted_views)
            
        except Exception as e:
            self.logger.warning(f"Performance prediction failed: {e}")
            return None
    
    def _rank_titles(self, titles: List[TitleResult]) -> List[TitleResult]:
        """Rank titles by overall score"""
        
        # Calculate composite score for ranking
        for title in titles:
            composite_score = (
                title.quality_score * 0.6 +
                title.confidence * 0.3 +
                (title.performance_prediction / 100000 if title.performance_prediction else 0.5) * 0.1
            )
            title.composite_score = composite_score
        
        # Sort by composite score
        return sorted(titles, key=lambda t: getattr(t, 'composite_score', 0), reverse=True)
    
    # Helper methods
    def _count_superlatives(self, title: str) -> int:
        """Count superlative words in title"""
        superlatives = ['best', 'worst', 'most', 'least', 'biggest', 'smallest', 
                       'greatest', 'ultimate', 'top', 'perfect', 'incredible']
        return sum(1 for word in superlatives if word in title.lower())
    
    def _identify_emotional_triggers(self, title: str) -> List[str]:
        """Identify emotional triggers in title"""
        triggers = {
            'curiosity': ['secret', 'hidden', 'mystery', 'revealed'],
            'urgency': ['now', 'finally', 'never', 'always'],
            'surprise': ['shocking', 'amazing', 'incredible', 'unbelievable'],
            'intensity': ['extreme', 'ultimate', 'massive', 'epic']
        }
        
        found_triggers = []
        title_lower = title.lower()
        
        for trigger_type, words in triggers.items():
            if any(word in title_lower for word in words):
                found_triggers.append(trigger_type)
        
        return found_triggers
    
    def _compare_length_to_optimal(self, title: str, channel_profile: ChannelProfile) -> Dict[str, Any]:
        """Compare title length to channel optimal"""
        title_length = len(title.split())
        optimal_length = channel_profile.title_patterns.avg_length_words
        
        return {
            'title_length': title_length,
            'optimal_length': optimal_length,
            'difference': abs(title_length - optimal_length),
            'is_optimal': abs(title_length - optimal_length) <= 1
        }
    
    def _calculate_pattern_adherence(self, 
                                   analysis: Dict[str, Any],
                                   channel_profile: ChannelProfile) -> float:
        """Calculate how well title adheres to successful channel patterns"""
        
        adherence_scores = []
        
        # Question format adherence
        if hasattr(channel_profile.title_patterns, 'question_ratio'):
            expected_questions = channel_profile.title_patterns.question_ratio > 0.3
            is_question = analysis['question_format']
            
            if expected_questions and is_question:
                adherence_scores.append(1.0)
            elif not expected_questions and not is_question:
                adherence_scores.append(1.0)
            else:
                adherence_scores.append(0.5)
        
        # Number usage adherence
        if hasattr(channel_profile.title_patterns, 'numeric_ratio'):
            expected_numbers = channel_profile.title_patterns.numeric_ratio > 0.3
            has_numbers = analysis['contains_numbers']
            
            if expected_numbers and has_numbers:
                adherence_scores.append(1.0)
            elif not expected_numbers and not has_numbers:
                adherence_scores.append(1.0)
            else:
                adherence_scores.append(0.5)
        
        return sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.5
    
    def _score_title_length(self, title: str, channel_profile: ChannelProfile) -> float:
        """Score title based on length optimization"""
        title_length = len(title.split())
        optimal_length = channel_profile.title_patterns.avg_length_words
        
        difference = abs(title_length - optimal_length)
        
        if difference == 0:
            return 1.0
        elif difference <= 1:
            return 0.9
        elif difference <= 2:
            return 0.7
        else:
            return 0.5
    
    def _score_uniqueness(self, title: str) -> float:
        """Score title uniqueness/creativity"""
        # Simple uniqueness scoring (can be enhanced)
        
        # Penalize very common words
        common_words = ['the', 'best', 'top', 'amazing', 'incredible', 'ultimate']
        common_count = sum(1 for word in common_words if word in title.lower())
        
        if common_count == 0:
            return 1.0
        elif common_count <= 2:
            return 0.8
        else:
            return 0.6
    
    # Response creation methods
    def _create_channel_analysis_summary(self, channel_profile: ChannelProfile) -> Dict[str, Any]:
        """Create summary of channel analysis for response"""
        
        return {
            'channel_type': channel_profile.channel_type,
            'total_videos': channel_profile.stats.total_videos,
            'avg_views': channel_profile.stats.avg_views,
            'high_performer_threshold': channel_profile.stats.high_performer_threshold,
            'successful_patterns': {
                'avg_title_length': channel_profile.title_patterns.avg_length_words,
                'question_usage': channel_profile.title_patterns.question_ratio,
                'numeric_usage': channel_profile.title_patterns.numeric_ratio,
                'superlative_usage': channel_profile.title_patterns.superlative_ratio
            }
        }
    
    def _calculate_confidence_range(self, titles: List[TitleResult]) -> Dict[str, float]:
        """Calculate confidence statistics"""
        if not titles:
            return {'min': 0, 'max': 0, 'avg': 0}
        
        confidences = [t.confidence for t in titles]
        return {
            'min': min(confidences),
            'max': max(confidences),
            'avg': sum(confidences) / len(confidences)
        }
    
    def _summarize_pattern_usage(self, titles: List[TitleResult]) -> Dict[str, Any]:
        """Summarize pattern usage across generated titles"""
        
        total_titles = len(titles)
        if total_titles == 0:
            return {}
        
        question_count = sum(1 for t in titles if t.pattern_analysis.get('question_format', False))
        number_count = sum(1 for t in titles if t.pattern_analysis.get('contains_numbers', False))
        superlative_count = sum(1 for t in titles if t.pattern_analysis.get('superlatives', 0) > 0)
        
        return {
            'question_format_usage': question_count / total_titles,
            'numeric_usage': number_count / total_titles,
            'superlative_usage': superlative_count / total_titles,
            'avg_length': sum(t.pattern_analysis.get('word_count', 0) for t in titles) / total_titles
        }
    
    def _generate_usage_recommendations(self,
                                      titles: List[TitleResult],
                                      channel_profile: ChannelProfile) -> List[str]:
        """Generate recommendations for using the titles"""
        
        recommendations = []
        
        if not titles:
            return ["No titles generated to analyze"]
        
        # Best title recommendation
        best_title = titles[0]  # Already ranked
        recommendations.append(
            f"Recommended: Use '{best_title.title}' - highest overall score ({best_title.quality_score:.2f})"
        )
        
        # Pattern recommendations
        high_confidence_titles = [t for t in titles if t.confidence > 0.7]
        if high_confidence_titles:
            recommendations.append(
                f"High confidence options: {len(high_confidence_titles)} titles show strong pattern alignment"
            )
        
        # Performance prediction
        predicted_titles = [t for t in titles if t.performance_prediction]
        if predicted_titles:
            best_predicted = max(predicted_titles, key=lambda t: t.performance_prediction)
            recommendations.append(
                f"Highest predicted performance: '{best_predicted.title}' ({best_predicted.performance_prediction:,.0f} views)"
            )
        
        return recommendations
    
    def _create_fallback_profile(self, channel_id: str) -> ChannelProfile:
        """Create basic fallback profile when profile creation fails"""
        # This would need proper implementation with actual ChannelProfile structure
        # For now, return a basic mock
        pass
    
    def _create_error_response(self,
                             request: TitleRequest,
                             error_msg: str,
                             processing_time: float) -> Dict[str, Any]:
        """Create error response"""
        
        return {
            'titles': [],
            'error': error_msg,
            'channel_analysis': {},
            'similar_examples': [],
            'generation_insights': {
                'processing_time': processing_time,
                'error': True,
                'recommendations': ["Generation failed - check configuration and try again"]
            },
            'metadata': {
                'request_id': f"error_{int(time.time())}",
                'generation_timestamp': datetime.now().isoformat(),
                'channel_id': request.channel_id,
                'original_idea': request.idea,
                'titles_requested': request.n_titles,
                'titles_generated': 0
            }
        }
    
    def _update_generation_metrics(self, processing_time: float, success: bool):
        """Update generation metrics"""
        
        self.generation_metrics['total_generations'] += 1
        
        if success:
            self.generation_metrics['successful_generations'] += 1
        else:
            self.generation_metrics['failed_generations'] += 1
        
        # Update average processing time
        total = self.generation_metrics['total_generations']
        current_avg = self.generation_metrics['avg_generation_time']
        new_avg = ((current_avg * (total - 1)) + processing_time) / total
        self.generation_metrics['avg_generation_time'] = new_avg
    
    def get_generation_metrics(self) -> Dict[str, Any]:
        """Get generation metrics"""
        return self.generation_metrics.copy()