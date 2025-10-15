#!/usr/bin/env python3
"""
Simple API test for TitleCraft AI
Quick validation of API endpoints
"""
import requests
import time
import sys

def test_server_health():
    """Test if server is running and healthy"""
    print("=== Testing Server Health ===")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Health: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_title_generation():
    """Test title generation endpoint"""
    print("\n=== Testing Title Generation ===")
    try:
        payload = {
            "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
            "idea": "How modern warfare tactics evolved from historical battles"
        }
        
        response = requests.post("http://localhost:8000/generate", json=payload, timeout=30)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generated {len(result['titles'])} titles:")
            for i, title in enumerate(result['titles'], 1):
                print(f"   {i}. {title['title']}")
                if 'reasoning' in title:
                    print(f"      Reasoning: {title['reasoning'][:80]}...")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generate failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid input"""
    print("\n=== Testing Error Handling ===")
    try:
        # Test with invalid channel
        payload = {
            "channel_id": "INVALID_CHANNEL_ID",
            "idea": "Test idea"
        }
        
        response = requests.post("http://localhost:8000/generate", json=payload, timeout=30)
        
        if response.status_code in [200, 400, 404]:
            print(f"✅ Proper error handling: {response.status_code}")
            if response.status_code == 200:
                print("   Server provided fallback titles for invalid channel")
            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run simple API tests"""
    print("TitleCraft AI - Simple API Test")
    print("=" * 35)
    
    # Wait a moment for server to be ready
    time.sleep(1)
    
    tests = [
        ("Server Health", test_server_health),
        ("Title Generation", test_title_generation), 
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        results.append(test_func())
    
    # Summary
    print("\n" + "=" * 35)
    print("TEST SUMMARY")
    print("=" * 35)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All API tests passed!")
    else:
        print("⚠️  Some API functionality needs attention")
        print("💡 Make sure the server is running: python -m src.apis.app")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)