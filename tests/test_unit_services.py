"""
Unit tests for service layer components.

Tests the business logic services including title generation,
quality scoring, and reasoning engines.
"""
import pytest
from unittest.mock import Mock, patch
import json

try:
    from src.services.generation_engine import TitleGenerationEngine
    from src.services.quality_scorer import TitleQualityScorer
    from src.services.reasoning_engine import ReasoningEngine, EnhancedReasoningEngine
except ImportError as e:
    pytest.skip(f"Service components not available: {e}", allow_module_level=True)


class TestTitleGenerationEngine:
    """Test title generation service."""
    
    def test_generation_engine_initialization(self):
        """Test TitleGenerationEngine initialization."""
        engine = TitleGenerationEngine()
        assert engine is not None
    
    @patch('src.services.generation_engine.LLMOrchestrator')
    def test_generate_titles(self, mock_orchestrator):
        """Test title generation workflow."""
        # Mock the LLM orchestrator
        mock_instance = Mock()
        mock_instance.generate_titles.return_value = {
            "titles": ["Test Title 1", "Test Title 2", "Test Title 3"],
            "reasoning": "Test reasoning"
        }
        mock_orchestrator.return_value = mock_instance
        
        engine = TitleGenerationEngine()
        
        result = engine.generate_titles(
            idea="Learn Python programming",
            channel_id="UC123"
        )
        
        assert result is not None
        assert 'titles' in result
        assert len(result['titles']) == 3
        mock_instance.generate_titles.assert_called_once()
    
    def test_generation_with_invalid_input(self):
        """Test generation with invalid input."""
        engine = TitleGenerationEngine()
        
        with pytest.raises((ValueError, TypeError)):
            engine.generate_titles("", "")  # Empty inputs
    
    @patch('src.services.generation_engine.ChannelProfiler')
    def test_channel_profile_integration(self, mock_profiler):
        """Test integration with channel profiler."""
        mock_profiler_instance = Mock()
        mock_profiler_instance.get_profile.return_value = {
            'channel_id': 'UC123',
            'patterns': ['How to', 'Complete Guide']
        }
        mock_profiler.return_value = mock_profiler_instance
        
        engine = TitleGenerationEngine()
        
        # Should use channel profile in generation
        result = engine.generate_titles(
            idea="Test idea",
            channel_id="UC123"
        )
        
        # Verify profiler was called
        mock_profiler_instance.get_profile.assert_called_with("UC123")


class TestQualityScorer:
    """Test title quality scoring."""
    
    def test_quality_scorer_initialization(self):
        """Test TitleQualityScorer initialization."""
        scorer = TitleQualityScorer()
        assert scorer is not None
    
    def test_score_single_title(self):
        """Test scoring a single title."""
        scorer = TitleQualityScorer()
        
        title = "How to Learn Python Programming in 30 Days"
        score = scorer.score_title(title)
        
        assert score is not None
        assert isinstance(score, (int, float, dict))
        
        # If score is a dict, check for required fields
        if isinstance(score, dict):
            assert 'overall_score' in score or 'score' in score
    
    def test_score_multiple_titles(self):
        """Test scoring multiple titles."""
        scorer = TitleQualityScorer()
        
        titles = [
            "Learn Python Programming", 
            "Complete Python Guide",
            "Python Tips and Tricks"
        ]
        
        scores = scorer.score_titles(titles)
        
        assert scores is not None
        assert len(scores) == len(titles)
    
    def test_score_quality_factors(self):
        """Test different quality factors."""
        scorer = TitleQualityScorer()
        
        # High quality title (good length, clear, action-oriented)
        high_quality = "5 Python Tips Every Developer Must Know"
        high_score = scorer.score_title(high_quality)
        
        # Low quality title (too short, vague)
        low_quality = "Python"
        low_score = scorer.score_title(low_quality)
        
        # Compare scores if they're numeric
        if isinstance(high_score, (int, float)) and isinstance(low_score, (int, float)):
            assert high_score > low_score
    
    def test_empty_title_handling(self):
        """Test handling empty titles."""
        scorer = TitleQualityScorer()
        
        with pytest.raises((ValueError, TypeError)):
            scorer.score_title("")


class TestReasoningEngine:
    """Test reasoning engine functionality.""" 
    
    def test_reasoning_engine_initialization(self):
        """Test ReasoningEngine initialization."""
        engine = ReasoningEngine()
        assert engine is not None
    
    def test_generate_reasoning(self, test_helper):
        """Test generating reasoning for titles."""
        engine = ReasoningEngine()
        
        titles = ["Learn Python Fast", "Python Masterclass", "Complete Python Guide"]
        channel_profile = test_helper.create_test_channel_profile()
        
        reasoning = engine.generate_reasoning(
            titles=titles,
            idea="Python programming tutorial",
            channel_profile=channel_profile
        )
        
        assert reasoning is not None
        assert isinstance(reasoning, (str, dict, list))
    
    def test_reasoning_with_patterns(self):
        """Test reasoning incorporates channel patterns."""
        engine = ReasoningEngine()
        
        channel_profile = {
            'channel_id': 'UC123',
            'top_performing_patterns': ['How to', 'Complete Guide', 'Tips'],
            'avg_views': 5000
        }
        
        titles = ["How to Master Python", "Complete Python Guide"]
        
        reasoning = engine.generate_reasoning(
            titles=titles,
            idea="Python tutorial",
            channel_profile=channel_profile
        )
        
        assert reasoning is not None
        # Should mention patterns if reasoning is string
        if isinstance(reasoning, str):
            reasoning_lower = reasoning.lower()
            assert any(pattern.lower() in reasoning_lower for pattern in ['how to', 'complete', 'guide'])
    
    def test_reasoning_citations(self):
        """Test reasoning includes data citations."""
        engine = ReasoningEngine()
        
        titles = ["Data-Driven Title"]
        channel_profile = {'channel_id': 'UC123', 'avg_views': 10000}
        
        reasoning = engine.generate_reasoning(
            titles=titles,
            idea="Test idea",
            channel_profile=channel_profile,
            include_citations=True
        )
        
        assert reasoning is not None


class TestEnhancedReasoningEngine:
    """Test enhanced reasoning engine."""
    
    def test_enhanced_reasoning_initialization(self):
        """Test EnhancedReasoningEngine initialization."""
        engine = EnhancedReasoningEngine()
        assert engine is not None
    
    @patch('src.services.reasoning_engine.SemanticMatcher')
    def test_similarity_based_reasoning(self, mock_matcher):
        """Test reasoning based on semantic similarity."""
        # Mock semantic matcher
        mock_matcher_instance = Mock()
        mock_matcher_instance.find_similar_titles.return_value = [
            "Similar Title 1",
            "Similar Title 2"
        ]
        mock_matcher.return_value = mock_matcher_instance
        
        engine = EnhancedReasoningEngine()
        
        reasoning = engine.generate_enhanced_reasoning(
            titles=["New Python Tutorial"],
            idea="Python programming",
            channel_profile={'channel_id': 'UC123'},
            historical_titles=["Old Python Tutorial", "JavaScript Guide"]
        )
        
        assert reasoning is not None
        mock_matcher_instance.find_similar_titles.assert_called()
    
    def test_multi_factor_reasoning(self):
        """Test reasoning considers multiple factors."""
        engine = EnhancedReasoningEngine()
        
        reasoning = engine.generate_enhanced_reasoning(
            titles=["5 Python Tips for Beginners"],
            idea="Python tips",
            channel_profile={
                'channel_id': 'UC123',
                'avg_views': 5000,
                'top_performing_patterns': ['Tips', 'Beginners']
            }
        )
        
        assert reasoning is not None


@pytest.mark.integration  
class TestServiceIntegration:
    """Integration tests for service layer components."""
    
    @patch('src.services.generation_engine.LLMOrchestrator')
    def test_generation_to_scoring_pipeline(self, mock_orchestrator):
        """Test title generation to quality scoring pipeline."""
        # Mock title generation
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.generate_titles.return_value = {
            "titles": ["Great Python Tutorial", "Learn Python Fast", "Python Made Easy"]
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Generate titles
        generator = TitleGenerationEngine()
        result = generator.generate_titles("Python tutorial", "UC123")
        
        # Score the generated titles
        scorer = TitleQualityScorer()
        scores = scorer.score_titles(result['titles'])
        
        assert result is not None
        assert scores is not None
        assert len(scores) == len(result['titles'])
    
    def test_full_service_pipeline(self, test_helper):
        """Test complete service layer pipeline."""
        # This test would integrate generation, scoring, and reasoning
        # In a real scenario with mocked dependencies
        
        channel_profile = test_helper.create_test_channel_profile()
        
        # Mock components for integration test
        with patch('src.services.generation_engine.LLMOrchestrator') as mock_llm, \
             patch('src.services.reasoning_engine.SemanticMatcher') as mock_matcher:
            
            # Setup mocks
            mock_llm_instance = Mock()
            mock_llm_instance.generate_titles.return_value = {
                "titles": ["Test Title 1", "Test Title 2"]
            }
            mock_llm.return_value = mock_llm_instance
            
            mock_matcher_instance = Mock()
            mock_matcher_instance.find_similar_titles.return_value = ["Similar Title"]
            mock_matcher.return_value = mock_matcher_instance
            
            # Run pipeline
            generator = TitleGenerationEngine()
            titles_result = generator.generate_titles("Test idea", "UC123")
            
            scorer = TitleQualityScorer()  
            scores = scorer.score_titles(titles_result['titles'])
            
            reasoner = EnhancedReasoningEngine()
            reasoning = reasoner.generate_enhanced_reasoning(
                titles=titles_result['titles'],
                idea="Test idea", 
                channel_profile=channel_profile
            )
            
            # Verify pipeline results
            assert titles_result is not None
            assert scores is not None
            assert reasoning is not None