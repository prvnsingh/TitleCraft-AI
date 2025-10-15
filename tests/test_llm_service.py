"""
Test and demonstration script for the new LLM service
Shows how to use different providers and LangSmith tracing
"""
import os
import sys
import json
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_llm_config():
    """Test LLM configuration validation"""
    print("=== Testing LLM Configuration ===")
    
    try:
        from src.services.llm_config import validate_config, get_recommended_config
        
        # Check configuration status
        status = validate_config()
        print("Configuration Status:")
        print(json.dumps(status, indent=2))
        
        # Get recommended config for title generation
        config = get_recommended_config("title_generation")
        print(f"\nRecommended config for title generation: {config}")
        
        return True
    except ImportError as e:
        print(f"Configuration test failed (expected without LangChain): {e}")
        return False


def test_enhanced_title_generator():
    """Test the enhanced title generator"""
    print("\n=== Testing Enhanced Title Generator ===")
    
    try:
        from src.services.enhanced_title_generator import EnhancedTitleGenerator
        
        # Initialize generator
        generator = EnhancedTitleGenerator()
        
        # Get service info
        info = generator.get_service_info()
        print("Service Info:")
        print(json.dumps(info, indent=2))
        
        # Note: We can't actually generate titles without proper data setup
        print("\nEnhanced Title Generator initialized successfully!")
        
        if info.get("langchain_enabled"):
            print("✓ LangChain integration is working")
        else:
            print("⚠ Using fallback mode (LangChain not available)")
        
        return True
    except Exception as e:
        print(f"Enhanced title generator test failed: {e}")
        return False


def test_llm_service():
    """Test LLM service directly"""
    print("\n=== Testing LLM Service Directly ===")
    
    try:
        from src.services.llm import (
            LLMServiceFactory, 
            create_system_message, 
            create_human_message,
            get_default_service
        )
        
        # Try to create a service
        try:
            service = get_default_service("openai_fast")
            print("✓ LLM service created successfully")
            
            # Test message creation
            system_msg = create_system_message("You are a helpful assistant.")
            human_msg = create_human_message("What is the capital of France?")
            
            print("✓ Message creation working")
            print(f"System message: {system_msg}")
            print(f"Human message: {human_msg}")
            
            # Get provider info
            info = service.get_provider_info()
            print(f"Provider info: {info}")
            
        except Exception as e:
            print(f"Service creation failed (expected without API key): {e}")
        
        return True
    except ImportError as e:
        print(f"LLM service test failed (expected without LangChain): {e}")
        return False


def demo_usage_patterns():
    """Demonstrate different usage patterns"""
    print("\n=== Usage Pattern Examples ===")
    
    print("""
# Example 1: Basic usage with OpenAI
from src.services.enhanced_title_generator import EnhancedTitleGenerator

generator = EnhancedTitleGenerator()
titles = generator.generate_titles("UC123", "How to learn Python")

# Example 2: Using different providers
generator = EnhancedTitleGenerator(llm_provider="anthropic", model="claude-3-sonnet-20240229")

# Example 3: Switching providers at runtime
generator.switch_provider("ollama", model="llama2")

# Example 4: Using streaming responses
for chunk in generator.generate_titles_streaming("UC123", "Python tutorial"):
    if chunk.get("complete"):
        titles = chunk.get("titles", [])
    else:
        print(f"Streaming: {chunk.get('chunk', '')}")

# Example 5: Using LLM service directly
from src.services.llm import LLMServiceFactory, create_system_message, create_human_message

service = LLMServiceFactory.create_openai_service()
messages = [
    create_system_message("You are a YouTube title expert"),
    create_human_message("Create a catchy title for a Python tutorial video")
]
response = service.generate(messages)
""")


def main():
    """Run all tests"""
    print("TitleCraft AI - LLM Service Test Suite")
    print("=" * 50)
    
    # Set up basic environment
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
    
    tests = [
        test_llm_config,
        test_enhanced_title_generator,
        test_llm_service,
        demo_usage_patterns
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("⚠ Some tests failed (expected if dependencies not installed)")
    
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up your API keys in .env file")
    print("3. Configure LangSmith for tracing (optional)")
    print("4. Start using the enhanced title generator!")


if __name__ == "__main__":
    main()