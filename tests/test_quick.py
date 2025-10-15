#!/usr/bin/env python3
"""
Quick test runner for TitleCraft AI
Simplified version of comprehensive tests for quick validation
"""
import sys
import os
import requests
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestRunner:
    """Simple test runner with consistent output"""
    
    def __init__(self):
        self.results = []
        self.current_test = ""
    
    def run_test(self, name: str, test_func):
        """Run a single test and track results"""
        self.current_test = name
        print(f"\n📋 {name}")
        print("-" * 40)
        
        try:
            success = test_func()
            if success:
                print(f"✅ {name} - PASSED")
                self.results.append((name, True, None))
            else:
                print(f"❌ {name} - FAILED")
                self.results.append((name, False, "Test returned False"))
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
            self.results.append((name, False, str(e)))
    
    def summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)
        
        print(f"Tests: {passed}/{total} passed")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed:")
            for name, success, error in self.results:
                if not success:
                    print(f"   ❌ {name}: {error}")
        
        return passed == total


def test_data_loading():
    """Test data loading functionality"""
    try:
        from src.data_module.data_processor import DataLoader
        
        loader = DataLoader()
        assert len(loader.data) > 0, "No data loaded"
        assert 'channel_id' in loader.data.columns, "Missing channel_id column"
        
        # Test channel retrieval
        channel_id = "UC510QYlOlKNyhy_zdQxnGYw"
        channel_data = loader.get_channel_data(channel_id)
        assert len(channel_data) > 0, "No channel data"
        
        print(f"   ✅ Loaded {len(loader.data)} videos from {len(loader.data['channel_id'].unique())} channels")
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False


def test_channel_analysis():
    """Test channel analysis functionality"""
    try:
        from src.data_module.data_processor import DataLoader
        
        loader = DataLoader()
        channel_id = "UC510QYlOlKNyhy_zdQxnGYw"
        analysis = loader.analyze_channel(channel_id)
        
        assert analysis.channel_id == channel_id, "Wrong channel in analysis"
        assert analysis.total_videos > 0, "No videos analyzed"
        assert len(analysis.top_performers) > 0, "No top performers found"
        
        print(f"   ✅ Analyzed {analysis.total_videos} videos, found {len(analysis.top_performers)} top performers")
        return True
        
    except Exception as e:
        print(f"   ❌ Channel analysis failed: {e}")
        return False


def test_title_generation():
    """Test title generation (fallback mode)"""
    try:
        from src.services.title_generator import TitleGenerator
        from src.data_module.data_processor import DataLoader
        
        # Test with no API key (fallback mode)
        original_key = os.environ.get('OPENAI_API_KEY')
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        try:
            # Create generator with fallback setup
            generator = TitleGenerator.__new__(TitleGenerator)
            generator.data_loader = DataLoader()
            generator.client = None
            
            titles = generator.generate_titles_fallback(
                channel_id="UC510QYlOlKNyhy_zdQxnGYw",
                idea="Modern warfare evolution",
                n_titles=3
            )
            
            assert len(titles) == 3, f"Expected 3 titles, got {len(titles)}"
            
            for title in titles:
                assert title.title, "Empty title generated"
                assert title.reasoning, "No reasoning provided"
            
            print(f"   ✅ Generated {len(titles)} titles with fallback mode")
            return True
            
        finally:
            # Restore API key if it existed
            if original_key:
                os.environ['OPENAI_API_KEY'] = original_key
        
    except Exception as e:
        print(f"   ❌ Title generation failed: {e}")
        return False


def test_api_models():
    """Test API model structure"""
    try:
        from src.apis.app import TitleRequest, TitleResponse, GenerationResponse
        
        # Test request model
        request = TitleRequest(
            channel_id="UC510QYlOlKNyhy_zdQxnGYw",
            idea="Test idea"
        )
        assert request.channel_id == "UC510QYlOlKNyhy_zdQxnGYw"
        
        # Test response models
        title_resp = TitleResponse(title="Test", reasoning="Test reasoning")
        gen_resp = GenerationResponse(
            titles=[title_resp],
            channel_id="test",
            idea="test"
        )
        assert len(gen_resp.titles) == 1
        
        print("   ✅ API models working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ API models failed: {e}")
        return False


def test_api_endpoints_offline():
    """Test API endpoints using TestClient"""
    try:
        from fastapi.testclient import TestClient
        from src.apis.app import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200, "Health endpoint failed"
        
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200, "Root endpoint failed"
        
        # Test generate endpoint
        response = client.post("/generate", json={
            "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
            "idea": "Test idea"
        })
        assert response.status_code == 200, f"Generate failed: {response.text}"
        
        data = response.json()
        assert "titles" in data, "Missing titles in response"
        assert len(data["titles"]) > 0, "No titles generated"
        
        print(f"   ✅ All endpoints working, generated {len(data['titles'])} titles")
        return True
        
    except Exception as e:
        print(f"   ❌ API endpoints failed: {e}")
        return False


def test_api_endpoints_live():
    """Test API endpoints with live server (optional)"""
    try:
        # Check if server is running
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("   ⏭️  Server not running, skipping live tests")
            return True
        
        # Test generate endpoint
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "channel_id": "UC510QYlOlKNyhy_zdQxnGYw", 
                "idea": "Modern warfare tactics"
            },
            timeout=30
        )
        
        assert response.status_code == 200, f"Live generate failed: {response.text}"
        
        data = response.json()
        assert "titles" in data, "Missing titles in live response"
        
        print(f"   ✅ Live server working, generated {len(data['titles'])} titles")
        return True
        
    except requests.RequestException:
        print("   ⏭️  Server not accessible, skipping live tests")
        return True
    except Exception as e:
        print(f"   ❌ Live API test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases"""
    try:
        from fastapi.testclient import TestClient
        from src.apis.app import app
        
        client = TestClient(app)
        
        # Test invalid request
        response = client.post("/generate", json={})
        assert response.status_code != 200, "Should reject empty request"
        
        # Test invalid channel (should handle gracefully)
        response = client.post("/generate", json={
            "channel_id": "INVALID_CHANNEL_123",
            "idea": "Test idea"
        })
        # Should either work with fallback or return proper error
        assert response.status_code in [200, 400, 404], f"Unexpected status: {response.status_code}"
        
        print("   ✅ Error handling working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False


def main():
    """Run quick test suite"""
    print("TitleCraft AI - Quick Test Suite")
    print("=" * 50)
    
    runner = TestRunner()
    
    # Run tests in logical order
    runner.run_test("Data Loading", test_data_loading)
    runner.run_test("Channel Analysis", test_channel_analysis)
    runner.run_test("Title Generation", test_title_generation)
    runner.run_test("API Models", test_api_models)
    runner.run_test("API Endpoints (Offline)", test_api_endpoints_offline)
    runner.run_test("API Endpoints (Live)", test_api_endpoints_live)
    runner.run_test("Error Handling", test_error_handling)
    
    return runner.summary()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)