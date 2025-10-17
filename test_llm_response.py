"""
Test script to see raw LLM response
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.title_generator import TitleGenerator, TitleGenerationRequest

# Create a test request
request = TitleGenerationRequest(
    channel_id="UC510QYlOlKNyhy_zdQxnGYw",
    video_idea="AI advances in 2024",
    n_titles=2,
    temperature=0.7
)

# Create title generator
generator = TitleGenerator()

# Generate titles and see what happens
try:
    result = generator.generate_titles(request)
    
    print("=== RESPONSE ===")
    print(f"Success: {result.success}")
    print(f"Titles generated: {len(result.titles)}")
    
    for i, title in enumerate(result.titles, 1):
        print(f"\nTitle {i}:")
        print(f"  Title: {title.title}")
        print(f"  Reasoning: {title.reasoning}")
        print(f"  Confidence: {title.confidence_score}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()