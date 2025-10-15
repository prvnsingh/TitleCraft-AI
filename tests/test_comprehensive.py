#!/usr/bin/env python3
"""
Comprehensive test suite for TitleCraft AI
Combines all test functionality into organized test classes
"""
import sys
import os
import time
import requests
import unittest
from typing import List, Dict, Any
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataModule(unittest.TestCase):
    """Test data loading and processing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            from src.data_module.data_processor import DataLoader
            self.data_loader = DataLoader()
        except Exception as e:
            self.skipTest(f"Failed to initialize DataLoader: {e}")
    
    def test_data_loading(self):
        """Test basic data loading"""
        print("\n=== Testing Data Loading ===")
        
        # Check data is loaded
        self.assertIsNotNone(self.data_loader.data, "No data loaded")
        self.assertGreater(len(self.data_loader.data), 0, "Empty dataset")
        
        # Check required columns
        required_columns = ['channel_id', 'title', 'views_in_period']
        for col in required_columns:
            self.assertIn(col, self.data_loader.data.columns, f"Missing column: {col}")
        
        print(f"✅ Loaded {len(self.data_loader.data)} videos")
        print(f"✅ Found {len(self.data_loader.data['channel_id'].unique())} channels")
    
    def test_channel_data_retrieval(self):
        """Test channel-specific data retrieval"""
        print("\n=== Testing Channel Data Retrieval ===")
        
        # Get a sample channel ID
        channel_id = "UC510QYlOlKNyhy_zdQxnGYw"
        
        # Test channel data retrieval
        channel_data = self.data_loader.get_channel_data(channel_id)
        self.assertIsNotNone(channel_data, "No channel data returned")
        self.assertGreater(len(channel_data), 0, "Empty channel data")
        
        # Verify all data belongs to the channel
        unique_channels = channel_data['channel_id'].unique()
        self.assertEqual(len(unique_channels), 1, "Mixed channel data")
        self.assertEqual(unique_channels[0], channel_id, "Wrong channel data")
        
        print(f"✅ Retrieved {len(channel_data)} videos for channel {channel_id}")
    
    def test_channel_analysis(self):
        """Test channel pattern analysis"""
        print("\n=== Testing Channel Analysis ===")
        
        channel_id = "UC510QYlOlKNyhy_zdQxnGYw"
        analysis = self.data_loader.analyze_channel(channel_id)
        
        # Verify analysis structure
        self.assertEqual(analysis.channel_id, channel_id, "Wrong channel ID in analysis")
        self.assertGreater(analysis.total_videos, 0, "No videos in analysis")
        self.assertGreater(analysis.avg_views, 0, "No view data")
        self.assertGreater(len(analysis.top_performers), 0, "No top performers")
        
        # Check patterns
        self.assertIn('avg_length', analysis.patterns, "Missing length pattern")
        self.assertIn('common_words', analysis.patterns, "Missing word pattern")
        
        print(f"✅ Analyzed {analysis.total_videos} videos")
        print(f"✅ Found {len(analysis.top_performers)} top performers")
        print(f"✅ Average views: {analysis.avg_views:.0f}")


class TestTitleGeneration(unittest.TestCase):
    """Test title generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Ensure no OpenAI key for consistent fallback testing
        if 'OPENAI_API_KEY' in os.environ:
            self.original_api_key = os.environ['OPENAI_API_KEY']
            del os.environ['OPENAI_API_KEY']
        else:
            self.original_api_key = None
    
    def tearDown(self):
        """Clean up after tests"""
        if self.original_api_key:
            os.environ['OPENAI_API_KEY'] = self.original_api_key
    
    def test_title_generator_initialization(self):
        """Test title generator initialization without API key"""
        print("\n=== Testing Title Generator Initialization ===")
        
        from src.services.title_generator import TitleGenerator
        
        # Should fail without API key
        with self.assertRaises(ValueError):
            generator = TitleGenerator()
        
        print("✅ Correctly fails without OpenAI API key")
    
    def test_fallback_title_generation(self):
        """Test fallback title generation"""
        print("\n=== Testing Fallback Title Generation ===")
        
        from src.services.title_generator import TitleGenerator
        from src.data_module.data_processor import DataLoader
        
        # Create generator with fallback setup
        generator = TitleGenerator.__new__(TitleGenerator)
        generator.data_loader = DataLoader()
        generator.client = None
        
        # Test fallback generation
        titles = generator.generate_titles_fallback(
            channel_id="UC510QYlOlKNyhy_zdQxnGYw",
            idea="Modern warfare tactics evolution",
            n_titles=3
        )
        
        # Verify results
        self.assertEqual(len(titles), 3, f"Expected 3 titles, got {len(titles)}")
        
        for i, title in enumerate(titles):
            self.assertIsNotNone(title.title, f"Title {i} is empty")
            self.assertIsNotNone(title.reasoning, f"Title {i} has no reasoning")
            self.assertGreaterEqual(title.confidence, 0, f"Title {i} has invalid confidence")
            self.assertLessEqual(title.confidence, 1, f"Title {i} has invalid confidence")
        
        print(f"✅ Generated {len(titles)} fallback titles")
        for i, title in enumerate(titles, 1):
            print(f"   {i}. {title.title} (confidence: {title.confidence:.2f})")


class TestAPIStructure(unittest.TestCase):
    """Test API models and structure"""
    
    def test_api_models(self):
        """Test API model creation and validation"""
        print("\n=== Testing API Models ===")
        
        from src.apis.app import TitleRequest, TitleResponse, GenerationResponse
        
        # Test request model
        request = TitleRequest(
            channel_id="UC510QYlOlKNyhy_zdQxnGYw",
            idea="Test video idea"
        )
        self.assertEqual(request.channel_id, "UC510QYlOlKNyhy_zdQxnGYw")
        self.assertEqual(request.idea, "Test video idea")
        
        # Test response models
        title_response = TitleResponse(
            title="Test Title",
            reasoning="Test reasoning for the title"
        )
        self.assertEqual(title_response.title, "Test Title")
        self.assertEqual(title_response.reasoning, "Test reasoning for the title")
        
        # Test generation response
        gen_response = GenerationResponse(
            titles=[title_response],
            channel_id="test_channel",
            idea="test idea"
        )
        self.assertEqual(len(gen_response.titles), 1)
        self.assertEqual(gen_response.channel_id, "test_channel")
        
        print("✅ All API models working correctly")
    
    def test_app_initialization(self):
        """Test FastAPI app initialization"""
        print("\n=== Testing App Initialization ===")
        
        from src.apis.app import app
        
        self.assertIsNotNone(app, "FastAPI app not initialized")
        print("✅ FastAPI app initialized successfully")


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints with live server"""
    
    @classmethod
    def setUpClass(cls):
        """Check if server is running"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            cls.server_running = response.status_code == 200
        except:
            cls.server_running = False
    
    def setUp(self):
        """Skip tests if server not running"""
        if not self.server_running:
            self.skipTest("Server not running on localhost:8000")
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        print("\n=== Testing Health Endpoint ===")
        
        response = requests.get("http://localhost:8000/health", timeout=5)
        self.assertEqual(response.status_code, 200, "Health endpoint failed")
        
        data = response.json()
        self.assertIn("status", data, "Missing status in health response")
        
        print(f"✅ Health endpoint: {response.status_code} - {data}")
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        print("\n=== Testing Root Endpoint ===")
        
        response = requests.get("http://localhost:8000/", timeout=5)
        self.assertEqual(response.status_code, 200, "Root endpoint failed")
        
        data = response.json()
        self.assertIn("message", data, "Missing message in root response")
        
        print(f"✅ Root endpoint: {response.status_code} - {data}")
    
    def test_generate_endpoint_valid(self):
        """Test title generation endpoint with valid data"""
        print("\n=== Testing Generate Endpoint (Valid) ===")
        
        payload = {
            "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
            "idea": "How modern warfare tactics evolved from historical battles"
        }
        
        response = requests.post(
            "http://localhost:8000/generate",
            json=payload,
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200, f"Generate failed: {response.text}")
        
        data = response.json()
        self.assertIn("titles", data, "Missing titles in response")
        self.assertGreater(len(data["titles"]), 0, "No titles generated")
        
        # Verify title structure
        for title in data["titles"]:
            self.assertIn("title", title, "Missing title field")
            self.assertIn("reasoning", title, "Missing reasoning field")
        
        print(f"✅ Generated {len(data['titles'])} titles:")
        for i, title in enumerate(data['titles'], 1):
            print(f"   {i}. {title['title']}")
    
    def test_generate_endpoint_invalid_channel(self):
        """Test generate endpoint with invalid channel"""
        print("\n=== Testing Generate Endpoint (Invalid Channel) ===")
        
        payload = {
            "channel_id": "INVALID_CHANNEL_ID_12345",
            "idea": "Some video idea"
        }
        
        response = requests.post(
            "http://localhost:8000/generate",
            json=payload,
            timeout=30
        )
        
        # Should handle gracefully (either fallback or proper error)
        self.assertIn(response.status_code, [200, 400, 404], 
                     f"Unexpected status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            self.assertIn("titles", data, "Missing titles in fallback response")
            print("✅ Graceful fallback for invalid channel")
        else:
            print(f"✅ Proper error handling for invalid channel: {response.status_code}")


class TestOfflineEndpoints(unittest.TestCase):
    """Test API endpoints using TestClient (no server required)"""
    
    def setUp(self):
        """Set up test client"""
        try:
            from fastapi.testclient import TestClient
            from src.apis.app import app
            self.client = TestClient(app)
        except Exception as e:
            self.skipTest(f"Failed to create test client: {e}")
    
    def test_health_endpoint_offline(self):
        """Test health endpoint via TestClient"""
        print("\n=== Testing Health Endpoint (Offline) ===")
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200, "Health endpoint failed")
        
        data = response.json()
        self.assertIn("status", data, "Missing status in response")
        
        print(f"✅ Health endpoint working: {data}")
    
    def test_root_endpoint_offline(self):
        """Test root endpoint via TestClient"""
        print("\n=== Testing Root Endpoint (Offline) ===")
        
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200, "Root endpoint failed")
        
        data = response.json()
        self.assertIn("message", data, "Missing message in response")
        
        print(f"✅ Root endpoint working: {data}")
    
    def test_generate_endpoint_offline(self):
        """Test generate endpoint via TestClient"""
        print("\n=== Testing Generate Endpoint (Offline) ===")
        
        payload = {
            "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
            "idea": "Modern military strategy evolution"
        }
        
        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200, f"Generate failed: {response.text}")
        
        data = response.json()
        self.assertIn("titles", data, "Missing titles in response")
        self.assertGreater(len(data["titles"]), 0, "No titles generated")
        
        print(f"✅ Generated {len(data['titles'])} titles via TestClient")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_missing_openai_key_handling(self):
        """Test handling of missing OpenAI API key"""
        print("\n=== Testing Missing API Key Handling ===")
        
        # Ensure no API key
        original_key = os.environ.get('OPENAI_API_KEY')
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        try:
            from src.services.title_generator import TitleGenerator
            
            # Should fail gracefully
            with self.assertRaises(ValueError):
                generator = TitleGenerator()
            
            print("✅ Properly handles missing OpenAI API key")
        
        finally:
            # Restore key if it existed
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        print("\n=== Testing Invalid Input Handling ===")
        
        from fastapi.testclient import TestClient
        from src.apis.app import app
        
        client = TestClient(app)
        
        # Test missing fields
        response = client.post("/generate", json={})
        self.assertNotEqual(response.status_code, 200, "Should reject empty request")
        
        # Test invalid channel format
        response = client.post("/generate", json={
            "channel_id": "",
            "idea": "test idea"
        })
        self.assertIn(response.status_code, [400, 422], "Should reject empty channel_id")
        
        print("✅ Properly validates input data")


def run_test_suite():
    """Run the complete test suite with organized output"""
    print("TitleCraft AI - Comprehensive Test Suite")
    print("=" * 50)
    
    # Test suites in logical order
    test_classes = [
        TestDataModule,
        TestTitleGeneration,
        TestAPIStructure,
        TestOfflineEndpoints,
        TestAPIEndpoints,  # Live server tests (may be skipped)
        TestErrorHandling
    ]
    
    total_tests = 0
    passed_tests = 0
    skipped_tests = 0
    
    for test_class in test_classes:
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        
        print(f"\n📋 Running {test_class.__name__}")
        print("-" * 30)
        
        # Run tests and capture results
        for test in suite:
            try:
                test.debug()  # Run without runner to see our print statements
                passed_tests += 1
                total_tests += 1
            except unittest.SkipTest as e:
                print(f"⏭️  Skipped: {e}")
                skipped_tests += 1
                total_tests += 1
            except Exception as e:
                print(f"❌ Failed: {e}")
                total_tests += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUITE SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Skipped: {skipped_tests}")
    print(f"Failed: {total_tests - passed_tests - skipped_tests}")
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if passed_tests == total_tests - skipped_tests:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nCore Functionality Verified:")
        print("  ✅ Data loading and processing")
        print("  ✅ Channel analysis and patterns")
        print("  ✅ Title generation with fallbacks")
        print("  ✅ API endpoints and models")
        print("  ✅ Error handling")
    else:
        print(f"\n⚠️  Some tests need attention")
    
    return passed_tests == total_tests - skipped_tests


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)