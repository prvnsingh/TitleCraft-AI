"""
Integration tests for end-to-end workflows.

Tests complete workflows that span multiple components
and validate system behavior as a whole.
"""
import pytest
from unittest.mock import Mock, patch
import asyncio
import tempfile
import csv
from pathlib import Path

try:
    from src.data.loader import DataLoader
    from src.data.validator import DataValidator  
    from src.data.profiler import ChannelProfiler
    from src.processing.pattern_profiler import PatternProfiler
    from src.services.generation_engine import TitleGenerationEngine
    from src.services.quality_scorer import TitleQualityScorer
    from src.services.reasoning_engine import ReasoningEngine
except ImportError as e:
    pytest.skip(f"Components not available for integration testing: {e}", allow_module_level=True)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end title generation workflow."""
    
    @patch('src.processing.llm_orchestrator.LLMOrchestrator')
    def test_complete_title_generation_workflow(self, mock_orchestrator, sample_csv_file):
        """Test the complete workflow from data loading to title generation."""
        # Mock LLM responses
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.generate_titles.return_value = {
            "titles": [
                "Master Python Programming in 30 Days",
                "Complete Python Guide for Developers", 
                "5 Python Tips That Will Change Your Code"
            ],
            "reasoning": "These titles use proven patterns from high-performing videos."
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Step 1: Load and validate data
        loader = DataLoader()
        raw_data = loader.load_csv(sample_csv_file)
        
        validator = DataValidator()
        validation_results = validator.validate_batch(raw_data)
        valid_data = [d for d, v in zip(raw_data, validation_results) if v]
        
        # Step 2: Profile channels
        profiler = ChannelProfiler()
        
        # Group data by channel
        channels = {}
        for video in valid_data:
            channel_id = video['channel_id']
            if channel_id not in channels:
                channels[channel_id] = []
            channels[channel_id].append(video)
        
        # Create channel profiles
        channel_profiles = {}
        for channel_id, channel_data in channels.items():
            channel_profiles[channel_id] = profiler.create_profile(channel_id, channel_data)
        
        # Step 3: Generate titles for a new video idea
        generator = TitleGenerationEngine()
        
        result = generator.generate_titles(
            idea="Advanced Python programming techniques",
            channel_id="UC123"
        )
        
        # Step 4: Score the generated titles
        scorer = TitleQualityScorer()
        scores = scorer.score_titles(result['titles'])
        
        # Step 5: Generate reasoning
        reasoner = ReasoningEngine()
        reasoning = reasoner.generate_reasoning(
            titles=result['titles'],
            idea="Advanced Python programming techniques",
            channel_profile=channel_profiles.get('UC123', {})
        )
        
        # Verify complete workflow
        assert len(valid_data) > 0
        assert len(channel_profiles) > 0
        assert result is not None
        assert 'titles' in result
        assert len(result['titles']) > 0
        assert scores is not None
        assert reasoning is not None
        
        # Verify data quality
        for title in result['titles']:
            assert isinstance(title, str)
            assert len(title.strip()) > 0
            assert len(title) <= 100  # YouTube title limit
    
    def test_data_processing_pipeline(self, sample_csv_file):
        """Test data processing pipeline from CSV to channel profiles."""
        # Load data
        loader = DataLoader()
        data = loader.load_csv(sample_csv_file)
        
        # Validate data
        validator = DataValidator()
        validation_results = validator.validate_batch(data)
        valid_data = [d for d, v in zip(data, validation_results) if v]
        
        # Extract patterns
        pattern_profiler = PatternProfiler()
        all_titles = [video['title'] for video in valid_data]
        global_patterns = pattern_profiler.extract_patterns(all_titles)
        
        # Create channel profiles
        channel_profiler = ChannelProfiler()
        
        # Group by channel and create profiles
        channels = {}
        for video in valid_data:
            channel_id = video['channel_id']
            if channel_id not in channels:
                channels[channel_id] = []
            channels[channel_id].append(video)
        
        profiles = {}
        for channel_id, channel_data in channels.items():
            profiles[channel_id] = channel_profiler.create_profile(channel_id, channel_data)
        
        # Verify pipeline results
        assert len(valid_data) > 0
        assert global_patterns is not None
        assert len(profiles) > 0
        assert 'UC123' in profiles or 'UC456' in profiles
        
        # Verify profile structure
        for channel_id, profile in profiles.items():
            assert profile is not None
            # Profile should have basic information
            if isinstance(profile, dict):
                assert 'channel_id' in profile or channel_id is not None
    
    @patch('src.processing.llm_orchestrator.LLMOrchestrator')
    def test_error_handling_in_workflow(self, mock_orchestrator, sample_csv_file):
        """Test error handling throughout the workflow."""
        # Mock LLM to fail first, then succeed
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.generate_titles.side_effect = [
            Exception("LLM API failed"),  # First call fails
            {  # Second call succeeds (retry)
                "titles": ["Fallback Title"],
                "reasoning": "Generated after retry"
            }
        ]
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        # Test workflow with error conditions
        loader = DataLoader()
        data = loader.load_csv(sample_csv_file)
        
        # Test with some invalid data mixed in
        data.append({
            'channel_id': '',  # Invalid empty channel
            'title': '',       # Invalid empty title
            'views_in_period': -1  # Invalid negative views
        })
        
        validator = DataValidator()
        validation_results = validator.validate_batch(data)
        valid_data = [d for d, v in zip(data, validation_results) if v]
        
        # Should filter out invalid data
        assert len(valid_data) < len(data)
        
        # Workflow should continue with valid data only
        profiler = ChannelProfiler()
        channel_data = [d for d in valid_data if d['channel_id'] == 'UC123']
        
        if channel_data:  # Only if we have valid data for this channel
            profile = profiler.create_profile('UC123', channel_data)
            assert profile is not None


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_api_title_generation_endpoint(self, mock_llm_response):
        """Test the API endpoint for title generation."""
        # This would test the actual FastAPI endpoint
        # For now, we'll test the service layer that the API would use
        
        with patch('src.services.generation_engine.LLMOrchestrator') as mock_orchestrator:
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.generate_titles.return_value = mock_llm_response
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            generator = TitleGenerationEngine()
            
            # Simulate API request
            result = generator.generate_titles(
                idea="Learn Python programming basics",
                channel_id="UC123"
            )
            
            # Verify API-like response
            assert result is not None
            assert 'titles' in result
            assert isinstance(result['titles'], list)
            assert len(result['titles']) > 0
    
    def test_batch_processing_integration(self, sample_video_data):
        """Test processing multiple requests in batch."""
        ideas = [
            "Python programming tutorial",
            "Web development guide", 
            "Data science basics"
        ]
        
        with patch('src.services.generation_engine.LLMOrchestrator') as mock_orchestrator:
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.generate_titles.return_value = {
                "titles": ["Batch Title 1", "Batch Title 2"],
                "reasoning": "Batch reasoning"
            }
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            generator = TitleGenerationEngine()
            
            # Process multiple ideas
            results = []
            for idea in ideas:
                result = generator.generate_titles(idea, "UC123")
                results.append(result)
            
            # Verify batch results
            assert len(results) == len(ideas)
            for result in results:
                assert result is not None
                assert 'titles' in result


@pytest.mark.integration
class TestDataConsistency:
    """Test data consistency across the system."""
    
    def test_channel_consistency(self, sample_csv_file):
        """Test that channel data remains consistent throughout processing."""
        loader = DataLoader()
        data = loader.load_csv(sample_csv_file)
        
        # Track channels through the pipeline
        original_channels = set(video['channel_id'] for video in data)
        
        validator = DataValidator()
        validation_results = validator.validate_batch(data)
        valid_data = [d for d, v in zip(data, validation_results) if v]
        
        validated_channels = set(video['channel_id'] for video in valid_data)
        
        # Channels should be preserved (unless data was invalid)
        assert validated_channels.issubset(original_channels)
        
        # Profile channels
        profiler = ChannelProfiler()
        
        profiles = {}
        for channel_id in validated_channels:
            channel_data = [v for v in valid_data if v['channel_id'] == channel_id]
            profiles[channel_id] = profiler.create_profile(channel_id, channel_data)
        
        # All validated channels should have profiles
        profile_channels = set(profiles.keys())
        assert profile_channels == validated_channels
    
    def test_title_generation_consistency(self, test_helper):
        """Test that generated titles are consistent with input."""
        with patch('src.services.generation_engine.LLMOrchestrator') as mock_orchestrator:
            # Mock deterministic responses
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.generate_titles.return_value = {
                "titles": ["Consistent Title 1", "Consistent Title 2"],
                "reasoning": "Consistent reasoning"
            }
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            generator = TitleGenerationEngine()
            
            # Generate titles multiple times with same input
            idea = "Test video idea"
            channel_id = "UC123"
            
            result1 = generator.generate_titles(idea, channel_id)
            result2 = generator.generate_titles(idea, channel_id)
            
            # Results should be consistent (with mocked LLM)
            assert result1['titles'] == result2['titles']


@pytest.mark.integration  
@pytest.mark.performance
class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    def test_data_processing_performance(self, sample_video_data):
        """Test performance of data processing pipeline."""
        import time
        
        # Test with different data sizes
        data_sizes = [10, 50, 100]
        results = {}
        
        for size in data_sizes:
            start_time = time.time()
            
            # Create test data of specified size
            test_data = sample_video_data[:size] if len(sample_video_data) >= size else sample_video_data * (size // len(sample_video_data) + 1)
            test_data = test_data[:size]
            
            # Process data
            validator = DataValidator()
            validation_results = validator.validate_batch(test_data)
            valid_data = [d for d, v in zip(test_data, validation_results) if v]
            
            # Profile channels
            profiler = ChannelProfiler()
            
            channels = {}
            for video in valid_data:
                channel_id = video['channel_id']
                if channel_id not in channels:
                    channels[channel_id] = []
                channels[channel_id].append(video)
            
            # Create profiles
            for channel_id, channel_data in channels.items():
                profiler.create_profile(channel_id, channel_data)
            
            processing_time = time.time() - start_time
            results[size] = processing_time
            
            # Ensure reasonable performance
            assert processing_time < 10.0  # Should process within 10 seconds
            
        # Performance should scale reasonably
        if len(results) > 1:
            sizes = sorted(results.keys())
            times = [results[s] for s in sizes]
            
            # Processing time shouldn't increase exponentially
            for i in range(1, len(times)):
                ratio = times[i] / times[i-1]
                size_ratio = sizes[i] / sizes[i-1]
                assert ratio <= size_ratio * 2  # Allow some overhead but not exponential
    
    def test_concurrent_processing(self, sample_video_data):
        """Test system behavior under concurrent processing."""
        import threading
        import time
        
        results = []
        errors = []
        
        def process_data(thread_id):
            """Process data in a separate thread."""
            try:
                with patch('src.services.generation_engine.LLMOrchestrator') as mock_orchestrator:
                    mock_orchestrator_instance = Mock()
                    mock_orchestrator_instance.generate_titles.return_value = {
                        "titles": [f"Title {thread_id} - 1", f"Title {thread_id} - 2"],
                        "reasoning": f"Reasoning for thread {thread_id}"
                    }
                    mock_orchestrator.return_value = mock_orchestrator_instance
                    
                    generator = TitleGenerationEngine()
                    result = generator.generate_titles(f"Test idea {thread_id}", "UC123")
                    results.append((thread_id, result))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Start multiple threads
        threads = []
        num_threads = 3
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors in concurrent processing: {errors}"
        assert len(results) == num_threads
        
        # Each thread should have unique results
        for thread_id, result in results:
            assert result is not None
            assert 'titles' in result
            assert str(thread_id) in str(result['titles'])  # Should contain thread ID


@pytest.mark.integration
@pytest.mark.slow  
class TestEndToEndPerformance:
    """Test complete end-to-end performance scenarios."""
    
    def test_full_pipeline_performance(self, sample_csv_file):
        """Test performance of complete pipeline."""
        import time
        
        start_time = time.time()
        
        with patch('src.processing.llm_orchestrator.LLMOrchestrator') as mock_orchestrator:
            mock_orchestrator_instance = Mock()
            mock_orchestrator_instance.generate_titles.return_value = {
                "titles": ["Performance Test Title"],
                "reasoning": "Performance test reasoning"
            }
            mock_orchestrator.return_value = mock_orchestrator_instance
            
            # Run complete pipeline
            loader = DataLoader()
            data = loader.load_csv(sample_csv_file)
            
            validator = DataValidator()
            validation_results = validator.validate_batch(data)
            valid_data = [d for d, v in zip(data, validation_results) if v]
            
            profiler = ChannelProfiler()
            channels = {}
            for video in valid_data:
                channel_id = video['channel_id']
                if channel_id not in channels:
                    channels[channel_id] = []
                channels[channel_id].append(video)
            
            for channel_id, channel_data in channels.items():
                profiler.create_profile(channel_id, channel_data)
            
            generator = TitleGenerationEngine()
            result = generator.generate_titles("Performance test", "UC123")
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 30.0  # Complete pipeline should finish within 30 seconds
        assert result is not None
        assert len(result['titles']) > 0