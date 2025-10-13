"""
Shared test configuration and fixtures for TitleCraft AI tests.

This module provides common fixtures, utilities, and configuration
for all test modules using pytest.
"""
# Standard library imports
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

# Third-party imports
import pandas as pd
import pytest

# Add src to Python path for all tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_video_data():
    """Sample video data for testing."""
    return [
        {
            "channel_id": "UC123",
            "video_id": "v1",
            "title": "How to Learn Python Programming",
            "summary": "Complete Python tutorial for beginners",
            "views_in_period": 5000,
            "upload_date": "2024-01-15"
        },
        {
            "channel_id": "UC123", 
            "video_id": "v2",
            "title": "Python for Beginners Guide",
            "summary": "Starting with Python programming basics",
            "views_in_period": 3000,
            "upload_date": "2024-01-20"
        },
        {
            "channel_id": "UC456",
            "video_id": "v3", 
            "title": "Cooking Italian Pasta",
            "summary": "Traditional pasta recipes from Italy",
            "views_in_period": 2000,
            "upload_date": "2024-01-25"
        }
    ]


@pytest.fixture
def sample_csv_file(temp_dir, sample_video_data):
    """Create a sample CSV file with test data."""
    csv_path = os.path.join(temp_dir, "test_data.csv")
    
    # Write CSV with proper headers
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['channel_id', 'video_id', 'title', 'summary', 'views_in_period', 'upload_date']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_video_data)
    
    return csv_path


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    return {
        "titles": [
            "5 Python Tips Every Developer Must Know",
            "Master Python Programming in 30 Days",
            "Python Secrets That Will Blow Your Mind"
        ],
        "reasoning": "These titles use strong emotional hooks and numbers to grab attention."
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[
            Mock(message=Mock(content='{"titles": ["Test Title 1", "Test Title 2"], "reasoning": "Test reasoning"}'))
        ]
    )
    return mock_client


@pytest.fixture
async def mock_cache_manager():
    """Mock cache manager for async testing."""
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = True
    cache.clear.return_value = True
    return cache


class TestHelper:
    """Helper utilities for tests."""
    
    @staticmethod
    def create_test_channel_profile(channel_id: str = "UC123") -> Dict[str, Any]:
        """Create a test channel profile."""
        return {
            "channel_id": channel_id,
            "total_videos": 50,
            "avg_views": 5000,
            "top_performing_patterns": [
                "How to",
                "Complete Guide",
                "Tips and Tricks"
            ],
            "common_topics": ["programming", "tutorials", "tips"],
            "title_length_stats": {
                "mean": 45.5,
                "median": 44.0,
                "std": 12.3
            }
        }
    
    @staticmethod
    def assert_valid_titles(titles: List[str]):
        """Assert that generated titles are valid."""
        assert isinstance(titles, list)
        assert len(titles) > 0
        for title in titles:
            assert isinstance(title, str)
            assert len(title.strip()) > 0
            assert len(title) <= 100  # YouTube title limit


@pytest.fixture
def test_helper():
    """Test helper utilities."""
    return TestHelper


# Pytest configuration functions
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests") 
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "slow: Slower tests that may be skipped in quick runs")
    config.addinivalue_line("markers", "data: Data layer tests")
    config.addinivalue_line("markers", "processing: Processing layer tests")
    config.addinivalue_line("markers", "services: Services layer tests")
    config.addinivalue_line("markers", "benchmark: Benchmark tests for performance measurement")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their file names."""
    for item in items:
        # Mark unit tests
        if "test_unit_" in item.fspath.basename:
            item.add_marker(pytest.mark.unit)
            
        # Mark integration tests
        if "test_integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
            
        # Mark performance tests
        if "test_performance" in item.fspath.basename:
            item.add_marker(pytest.mark.performance)
            
        # Mark data layer tests
        if "test_unit_data" in item.fspath.basename:
            item.add_marker(pytest.mark.data)
            
        # Mark processing layer tests  
        if "test_unit_processing" in item.fspath.basename:
            item.add_marker(pytest.mark.processing)
            
        # Mark services layer tests
        if "test_unit_services" in item.fspath.basename:
            item.add_marker(pytest.mark.services)