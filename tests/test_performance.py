"""
Performance tests for TitleCraft AI system.

Tests system performance, scalability, and resource usage
under various load conditions.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from src.data.loader import DataLoader
    from src.data.validator import DataValidator
    from src.data.profiler import ChannelProfiler
    from src.services.generation_engine import TitleGenerationEngine
    from src.services.quality_scorer import TitleQualityScorer
except ImportError as e:
    pytest.skip(f"Components not available for performance testing: {e}", allow_module_level=True)


@pytest.mark.performance
class TestDataProcessingPerformance:
    """Test performance of data processing components."""
    
    def test_data_loading_performance(self, sample_csv_file):
        """Test performance of data loading operations."""
        start_time = time.time()
        
        loader = DataLoader()
        data = loader.load_csv(sample_csv_file)
        
        load_time = time.time() - start_time
        
        # Performance assertions
        assert load_time < 5.0  # Should load within 5 seconds
        assert len(data) > 0
        
        # Test memory usage
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 500  # Should use less than 500MB
    
    def test_data_validation_performance(self, sample_video_data):
        """Test performance of data validation with different data sizes."""
        validator = DataValidator()
        
        # Test different sizes
        sizes = [10, 100, 1000]
        
        for size in sizes:
            # Create test data of specified size
            test_data = (sample_video_data * (size // len(sample_video_data) + 1))[:size]
            
            start_time = time.time()
            results = validator.validate_batch(test_data)
            validation_time = time.time() - start_time
            
            # Performance should scale linearly
            expected_max_time = size * 0.01  # 10ms per item max
            assert validation_time < expected_max_time
            assert len(results) == size
    
    def test_channel_profiling_performance(self, sample_video_data):
        """Test performance of channel profiling operations."""
        profiler = ChannelProfiler()
        
        # Group data by channel
        channels = {}
        for video in sample_video_data:
            channel_id = video['channel_id']
            if channel_id not in channels:
                channels[channel_id] = []
            channels[channel_id].append(video)
        
        profile_times = []
        
        for channel_id, channel_data in channels.items():
            start_time = time.time()
            profile = profiler.create_profile(channel_id, channel_data)
            profile_time = time.time() - start_time
            profile_times.append(profile_time)
            
            # Individual profile creation should be fast
            assert profile_time < 2.0
            assert profile is not None
        
        # Average profiling time should be reasonable
        avg_time = sum(profile_times) / len(profile_times) if profile_times else 0
        assert avg_time < 1.0


@pytest.mark.performance
class TestServicePerformance:
    """Test performance of service layer components."""
    
    @patch('src.processing.llm_orchestrator.LLMOrchestrator')
    def test_title_generation_performance(self, mock_orchestrator):
        """Test performance of title generation service."""
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.generate_titles.return_value = {
            "titles": ["Performance Test Title 1", "Performance Test Title 2"],
            "reasoning": "Performance test reasoning"
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        generator = TitleGenerationEngine()
        
        generation_times = []
        
        # Test multiple generations
        for i in range(10):
            start_time = time.time()
            result = generator.generate_titles(f"Test idea {i}", "UC123")
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            # Each generation should be reasonably fast
            assert generation_time < 5.0
            assert result is not None
            assert 'titles' in result
        
        # Average generation time should be consistent
        avg_time = sum(generation_times) / len(generation_times)
        assert avg_time < 2.0
        
        # Performance should be consistent (not degrading)
        first_half_avg = sum(generation_times[:5]) / 5
        second_half_avg = sum(generation_times[5:]) / 5
        
        # Second half shouldn't be significantly slower (allow 50% variance)
        assert second_half_avg < first_half_avg * 1.5
    
    def test_quality_scoring_performance(self, mock_llm_response):
        """Test performance of quality scoring operations."""
        scorer = TitleQualityScorer()
        
        # Test with different numbers of titles
        title_counts = [5, 20, 50]
        
        for count in title_counts:
            titles = [f"Test Title {i}" for i in range(count)]
            
            start_time = time.time()
            scores = scorer.score_titles(titles)
            scoring_time = time.time() - start_time
            
            # Scoring time should scale linearly
            max_expected_time = count * 0.1  # 100ms per title max
            assert scoring_time < max_expected_time
            assert scores is not None


@pytest.mark.performance
@pytest.mark.slow
class TestConcurrencyPerformance:
    """Test system performance under concurrent load."""
    
    @patch('src.processing.llm_orchestrator.LLMOrchestrator')
    def test_concurrent_title_generation(self, mock_orchestrator):
        """Test concurrent title generation performance."""
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.generate_titles.return_value = {
            "titles": ["Concurrent Title 1", "Concurrent Title 2"],
            "reasoning": "Concurrent reasoning"
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        generator = TitleGenerationEngine()
        
        def generate_title(thread_id):
            """Generate title in thread."""
            start_time = time.time()
            result = generator.generate_titles(f"Concurrent test {thread_id}", "UC123")
            generation_time = time.time() - start_time
            return thread_id, result, generation_time
        
        # Test with different numbers of concurrent threads
        thread_counts = [2, 5, 10]
        
        for num_threads in thread_counts:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(generate_title, i) 
                    for i in range(num_threads)
                ]
                
                results = []
                for future in as_completed(futures):
                    thread_id, result, gen_time = future.result()
                    results.append((thread_id, result, gen_time))
            
            total_time = time.time() - start_time
            
            # All threads should complete
            assert len(results) == num_threads
            
            # Total time shouldn't be much more than sequential
            # (allowing for some overhead)
            max_individual_time = max(gen_time for _, _, gen_time in results)
            assert total_time < max_individual_time * 2
            
            # All results should be valid
            for thread_id, result, gen_time in results:
                assert result is not None
                assert 'titles' in result
                assert gen_time < 10.0  # Individual operations should be fast
    
    def test_memory_usage_under_load(self, sample_video_data):
        """Test memory usage under concurrent load."""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        def process_data(thread_id):
            """Process data in thread."""
            validator = DataValidator()
            profiler = ChannelProfiler()
            
            # Validate data
            validation_results = validator.validate_batch(sample_video_data)
            valid_data = [d for d, v in zip(sample_video_data, validation_results) if v]
            
            # Create profiles
            channels = {}
            for video in valid_data:
                channel_id = video['channel_id']
                if channel_id not in channels:
                    channels[channel_id] = []
                channels[channel_id].append(video)
            
            profiles = {}
            for channel_id, channel_data in channels.items():
                profiles[channel_id] = profiler.create_profile(channel_id, channel_data)
            
            return len(profiles)
        
        # Run concurrent processing
        num_threads = 5
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(process_data, i) 
                for i in range(num_threads)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory usage shouldn't grow excessively
        assert memory_increase < 200  # Less than 200MB increase
        assert len(results) == num_threads
        assert all(r > 0 for r in results)  # All threads should produce results


@pytest.mark.performance
@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Benchmark tests for system scalability."""
    
    def test_data_size_scalability(self, sample_video_data):
        """Test how performance scales with data size."""
        validator = DataValidator()
        profiler = ChannelProfiler()
        
        # Test different data sizes
        sizes = [10, 50, 100, 200]
        performance_metrics = {}
        
        for size in sizes:
            # Create test data of specified size
            test_data = (sample_video_data * (size // len(sample_video_data) + 1))[:size]
            
            # Measure validation performance
            start_time = time.time()
            validation_results = validator.validate_batch(test_data)
            validation_time = time.time() - start_time
            
            # Measure profiling performance
            valid_data = [d for d, v in zip(test_data, validation_results) if v]
            
            profiling_start = time.time()
            channels = {}
            for video in valid_data:
                channel_id = video['channel_id']
                if channel_id not in channels:
                    channels[channel_id] = []
                channels[channel_id].append(video)
            
            for channel_id, channel_data in channels.items():
                profiler.create_profile(channel_id, channel_data)
                
            profiling_time = time.time() - profiling_start
            
            performance_metrics[size] = {
                'validation_time': validation_time,
                'profiling_time': profiling_time,
                'total_time': validation_time + profiling_time,
                'valid_records': len(valid_data)
            }
        
        # Analyze scalability
        sizes_list = sorted(performance_metrics.keys())
        
        for i in range(1, len(sizes_list)):
            current_size = sizes_list[i]
            previous_size = sizes_list[i-1]
            
            current_metrics = performance_metrics[current_size]
            previous_metrics = performance_metrics[previous_size]
            
            size_ratio = current_size / previous_size
            time_ratio = current_metrics['total_time'] / previous_metrics['total_time']
            
            # Time growth should be roughly linear (allow some overhead)
            # Time ratio shouldn't exceed size ratio by more than 50%
            assert time_ratio <= size_ratio * 1.5
            
        # Verify all operations completed successfully
        for size, metrics in performance_metrics.items():
            assert metrics['validation_time'] > 0
            assert metrics['profiling_time'] > 0
            assert metrics['valid_records'] > 0
    
    @patch('src.processing.llm_orchestrator.LLMOrchestrator')
    def test_throughput_benchmark(self, mock_orchestrator, sample_video_data):
        """Benchmark system throughput."""
        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.generate_titles.return_value = {
            "titles": ["Benchmark Title"],
            "reasoning": "Benchmark reasoning"
        }
        mock_orchestrator.return_value = mock_orchestrator_instance
        
        generator = TitleGenerationEngine()
        
        # Measure throughput over time
        duration = 5  # seconds
        start_time = time.time()
        generations = 0
        
        while time.time() - start_time < duration:
            result = generator.generate_titles("Throughput test", "UC123")
            assert result is not None
            generations += 1
        
        actual_duration = time.time() - start_time
        throughput = generations / actual_duration
        
        # System should handle reasonable throughput
        assert throughput > 0.1  # At least 0.1 generations per second
        assert generations > 0
        
        print(f"Throughput: {throughput:.2f} generations/second ({generations} in {actual_duration:.2f}s)")


@pytest.mark.performance
class TestResourceUsage:
    """Test system resource usage patterns."""
    
    def test_cpu_usage_monitoring(self, sample_video_data):
        """Monitor CPU usage during processing."""
        validator = DataValidator()
        
        # Monitor CPU before processing
        cpu_before = psutil.cpu_percent(interval=1)
        
        # Perform CPU-intensive operation
        large_data = sample_video_data * 10  # Make it larger
        
        cpu_start = time.time()
        validation_results = validator.validate_batch(large_data)
        cpu_end = time.time()
        
        # Monitor CPU after processing  
        cpu_after = psutil.cpu_percent(interval=1)
        
        processing_time = cpu_end - cpu_start
        
        # Verify processing completed
        assert len(validation_results) == len(large_data)
        assert processing_time > 0
        
        # CPU usage should be reasonable (not maxed out continuously)
        assert cpu_after < 90  # Less than 90% CPU usage
    
    def test_memory_leak_detection(self, sample_video_data):
        """Test for memory leaks in repeated operations."""
        validator = DataValidator()
        profiler = ChannelProfiler()
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_measurements = [initial_memory]
        
        # Perform repeated operations
        for i in range(10):
            # Process data
            validation_results = validator.validate_batch(sample_video_data)
            valid_data = [d for d, v in zip(sample_video_data, validation_results) if v]
            
            # Create profiles
            channels = {}
            for video in valid_data:
                channel_id = video['channel_id']
                if channel_id not in channels:
                    channels[channel_id] = []
                channels[channel_id].append(video)
            
            for channel_id, channel_data in channels.items():
                profiler.create_profile(channel_id, channel_data)
            
            # Measure memory
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_measurements.append(current_memory)
        
        final_memory = memory_measurements[-1]
        memory_growth = final_memory - initial_memory
        
        # Memory shouldn't grow excessively (allowing for some caching)
        assert memory_growth < 100  # Less than 100MB growth
        
        # Check for consistent memory growth (potential leak)
        if len(memory_measurements) >= 5:
            # Memory shouldn't increase consistently across all measurements
            increasing_count = 0
            for i in range(1, len(memory_measurements)):
                if memory_measurements[i] > memory_measurements[i-1]:
                    increasing_count += 1
            
            # Not all measurements should show increase (allow some fluctuation)
            assert increasing_count < len(memory_measurements) * 0.8
