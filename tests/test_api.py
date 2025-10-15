#!/usr/bin/env python3
"""
API endpoint testing script for TitleCraft AI
"""
import requests
import json
import time

def test_health_endpoint():
    """Test the health endpoint"""
    print("=== Testing Health Endpoint ===")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_generate_endpoint():
    """Test the title generation endpoint"""
    print("\n=== Testing Generate Endpoint ===")
    try:
        payload = {
            "channel_id": "UC510QYlOlKNyhy_zdQxnGYw",
            "idea": "How modern warfare tactics evolved from historical battles"
        }
        
        response = requests.post(
            "http://localhost:8000/generate", 
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generated {len(result['titles'])} titles:")
            for i, title_obj in enumerate(result['titles'], 1):
                print(f"   {i}. {title_obj['title']}")
                print(f"      Reasoning: {title_obj['reasoning'][:100]}...")
            return True
        else:
            print(f"❌ Generation failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generate endpoint failed: {e}")
        return False

def test_invalid_channel():
    """Test with invalid channel ID"""
    print("\n=== Testing Invalid Channel ID ===")
    try:
        payload = {
            "channel_id": "INVALID_CHANNEL_ID",
            "idea": "Some video idea"
        }
        
        response = requests.post(
            "http://localhost:8000/generate", 
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 404:
            print("✅ Correctly handled invalid channel ID")
            return True
        else:
            print(f"❌ Unexpected response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Invalid channel test failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("TitleCraft AI - API Endpoint Testing")
    print("=" * 40)
    
    # Wait for server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    results = []
    results.append(test_health_endpoint())
    results.append(test_root_endpoint())
    results.append(test_generate_endpoint())
    results.append(test_invalid_channel())
    
    print(f"\n=== API Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All API endpoints working correctly!")
    else:
        print("❌ Some API functionality issues detected")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)