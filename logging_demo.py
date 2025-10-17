"""
Test script to demonstrate comprehensive logging throughout TitleCraft AI

This script makes a complete title generation request to show the full logging flow
and demonstrate how logs provide insights into the code execution and reasoning.
"""

import time
from src.services.title_generator import TitleGenerator, TitleGenerationRequest
from src.services.logger_config import titlecraft_logger

def main():
    """Demonstrate the logging system with a complete title generation flow"""
    
    # Get the main logger
    main_logger = titlecraft_logger.get_logger("demo")
    
    print("=" * 80)
    print("TitleCraft AI - Complete Logging Flow Demonstration")
    print("=" * 80)
    
    main_logger.info("Starting TitleCraft AI logging demonstration", extra={
        'extra_fields': {
            'component': 'demo',
            'action': 'demo_start',
            'timestamp': time.time()
        }
    })
    
    try:
        # Initialize the title generator (this will trigger initialization logs)
        print("\n🚀 Initializing TitleGenerator...")
        generator = TitleGenerator()
        
        # Create a test request
        print("\n📝 Creating title generation request...")
        request = TitleGenerationRequest(
            channel_id="UC510QYlOlKNyhy_zdQxnGYw",  # Use one of the available channels
            video_idea="How to master Python programming with AI assistance",
            n_titles=3,
            model_name="DeepSeek-R1-Distill-Qwen-32B"
        )
        
        main_logger.info("Test request created", extra={
            'extra_fields': {
                'component': 'demo',
                'action': 'request_created',
                'video_idea': request.video_idea,
                'n_titles': request.n_titles
            },
            'channel_id': request.channel_id
        })
        
        # Generate titles (this will trigger the full logging flow)
        print("\n🤖 Generating titles with full logging...")
        print("   📊 Data analysis and pattern discovery")
        print("   🧠 Context-aware prompt selection")  
        print("   💬 LLM interaction and response processing")
        print("   🏆 Quality evaluation and ranking")
        
        response = generator.generate_titles(request)
        
        # Display results and log summary
        print("\n" + "="*60)
        print("📋 GENERATION RESULTS")
        print("="*60)
        
        if response.success:
            print(f"✅ Success! Generated {len(response.titles)} titles")
            print(f"⏱️  Response Time: {response.response_time:.2f}s")
            print(f"🤖 Model Used: {response.model_used}")
            print(f"💰 Estimated Cost: ${response.estimated_cost or 0:.4f}")
            print(f"📊 Tokens Used: {response.tokens_used or 'N/A'}")
            
            print("\n📝 Generated Titles:")
            for i, title in enumerate(response.titles, 1):
                print(f"\n{i}. {title.title}")
                print(f"   Confidence: {title.confidence_score:.1%}")
                print(f"   Reasoning: {title.reasoning[:100]}...")
                
            main_logger.info("Title generation demo completed successfully", extra={
                'extra_fields': {
                    'component': 'demo',
                    'action': 'demo_success',
                    'titles_generated': len(response.titles),
                    'response_time': response.response_time,
                    'model_used': response.model_used
                },
                'channel_id': request.channel_id,
                'request_id': response.request_id
            })
        else:
            print(f"❌ Generation failed: {response.error_message}")
            main_logger.error("Title generation demo failed", extra={
                'extra_fields': {
                    'component': 'demo',
                    'action': 'demo_failure',
                    'error_message': response.error_message
                },
                'channel_id': request.channel_id
            })
        
        print("\n" + "="*60)
        print("📁 LOG FILES CREATED")
        print("="*60)
        print("Check the following log files for detailed insights:")
        print("• logs/titlecraft_app.log - Main application flow")
        print("• logs/titlecraft_debug.log - Detailed debugging info")  
        print("• logs/titlecraft_flow.log - Code execution flow")
        print("• logs/titlecraft_performance.log - Performance metrics")
        
        print("\n💡 PRESENTATION INSIGHTS")
        print("="*60)
        print("The logs show:")
        print("1. 🔄 Complete request flow from API to response")
        print("2. 📊 Data analysis and pattern discovery process")
        print("3. 🧠 Context-aware decision making and reasoning")
        print("4. ⚡ Performance metrics and execution timing")
        print("5. 🎯 Quality evaluation and title ranking logic")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        main_logger.error("Demo execution failed", extra={
            'extra_fields': {
                'component': 'demo',
                'action': 'demo_error',
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
        })
        
    print("\n🎯 Demo completed! Check the log files for comprehensive execution details.")

if __name__ == "__main__":
    main()