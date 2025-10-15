#!/usr/bin/env python3
"""
Quick functionality test for TitleCraft AI
"""
import sys
import os

def test_data_loading():
    """Test data loading functionality"""
    print("=== Testing Data Loading ===")
    try:
        from src.data_module.data_processor import DataLoader
        loader = DataLoader()
        print(f"✅ Data loading successful!")
        print(f"   Channels available: {list(loader.data['channel_id'].unique())}")
        print(f"   Total videos: {len(loader.data)}")
        
        # Test channel analysis
        channel_id = "UC510QYlOlKNyhy_zdQxnGYw"
        channel_data = loader.get_channel_data(channel_id)
        print(f"   Videos for sample channel: {len(channel_data)}")
        
        # Test analysis
        analysis = loader.analyze_channel(channel_id)
        print(f"   Top performers: {len(analysis.top_performers)}")
        print(f"   Average views: {analysis.avg_views:.0f}")
        print(f"   Sample patterns: {analysis.patterns}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_title_generation_fallback():
    """Test title generation with fallback (no API key)"""
    print("\n=== Testing Title Generation (Fallback) ===")
    try:
        # Remove API key to test fallback
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
            
        from src.services.title_generator import TitleGenerator
        # This should fail gracefully and use fallback
        try:
            generator = TitleGenerator()
            print("❌ Should have failed without API key")
            return False
        except ValueError as e:
            print(f"✅ Correctly failed without API key: {e}")
            
        # Test fallback generation directly
        generator = TitleGenerator.__new__(TitleGenerator)
        generator.data_loader = None
        from src.data_module.data_processor import DataLoader
        generator.data_loader = DataLoader()
        
        fallback_titles = generator._generate_fallback_titles("modern warfare tactics", 3)
        print(f"✅ Fallback generation works: {len(fallback_titles)} titles")
        for i, title in enumerate(fallback_titles, 1):
            print(f"   {i}. {title.title}")
            
        return True
    except Exception as e:
        print(f"❌ Title generation test failed: {e}")
        return False

def test_api_structure():
    """Test API structure and models"""
    print("\n=== Testing API Structure ===")
    try:
        from src.apis.app import app, TitleRequest, TitleResponse, GenerationResponse
        print("✅ API imports successful")
        
        # Test model creation
        request = TitleRequest(
            channel_id="UC510QYlOlKNyhy_zdQxnGYw",
            idea="How modern warfare evolved"
        )
        print(f"✅ Request model works: {request.channel_id}")
        
        response = TitleResponse(
            title="Test Title",
            reasoning="Test reasoning"
        )
        print(f"✅ Response model works: {response.title}")
        
        return True
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("TitleCraft AI - Functionality Test")
    print("=" * 40)
    
    results = []
    results.append(test_data_loading())
    results.append(test_title_generation_fallback())
    results.append(test_api_structure())
    
    print(f"\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All core functionality working!")
    else:
        print("❌ Some functionality issues detected")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)