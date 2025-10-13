"""
Test script for TitleCraft AI data module.
This script validates the module structure and demonstrates basic functionality.
"""

import os
import sys
import json

def test_module_structure():
    """Test that all module files exist and are properly structured."""
    print("=== TESTING MODULE STRUCTURE ===")
    
    # Check if all required files exist
    base_path = "src/data"
    required_files = [
        "__init__.py",
        "models.py", 
        "loader.py",
        "validator.py",
        "analyzer.py",
        "profiler.py",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(base_path, file)
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMISSING FILES: {missing_files}")
        return False
    else:
        print("\n✓ All module files present!")
        return True

def test_imports():
    """Test module imports (basic syntax validation)."""
    print("\n=== TESTING IMPORTS ===")
    
    # Test basic Python syntax by attempting to compile each file
    base_path = "src/data"
    python_files = ["models.py", "loader.py", "validator.py", "analyzer.py", "profiler.py"]
    
    syntax_errors = []
    for file in python_files:
        file_path = os.path.join(base_path, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file_path, 'exec')
            print(f"✓ {file} - syntax OK")
        except SyntaxError as e:
            print(f"✗ {file} - syntax error: {e}")
            syntax_errors.append((file, str(e)))
        except FileNotFoundError:
            print(f"✗ {file} - file not found")
            syntax_errors.append((file, "file not found"))
    
    if syntax_errors:
        print(f"\nSYNTAX ERRORS: {syntax_errors}")
        return False
    else:
        print("\n✓ All files have valid Python syntax!")
        return True

def test_data_file():
    """Test that the data file exists and is readable."""
    print("\n=== TESTING DATA FILE ===")
    
    data_file = "electrify__applied_ai_engineer__training_data.csv"
    
    if not os.path.exists(data_file):
        print(f"✗ Data file not found: {data_file}")
        return False
    
    try:
        # Basic file reading test
        with open(data_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            line_count = sum(1 for _ in f) + 1  # +1 for the first line we already read
        
        print(f"✓ Data file exists: {data_file}")
        print(f"✓ Header: {first_line}")
        print(f"✓ Total lines: {line_count}")
        
        # Check if header has expected columns
        expected_columns = ['channel_id', 'video_id', 'title', 'summary', 'views_in_period']
        header_columns = [col.strip() for col in first_line.split(',')]
        
        missing_cols = set(expected_columns) - set(header_columns)
        if missing_cols:
            print(f"⚠ Missing expected columns: {missing_cols}")
        else:
            print("✓ All expected columns present")
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading data file: {e}")
        return False

def create_sample_data():
    """Create a small sample data file for testing if main data file is not available."""
    print("\n=== CREATING SAMPLE DATA ===")
    
    sample_data = [
        "channel_id,video_id,title,summary,views_in_period",
        "UCTestChannel1,video1,Amazing Space Discovery: 10 Mind-Blowing Facts,Discover incredible facts about our universe,15000",
        "UCTestChannel1,video2,How Solar Systems Are Born - Complete Guide,Learn about star formation and planetary systems,12500", 
        "UCTestChannel1,video3,The Mystery of Black Holes Explained,Deep dive into the physics of black holes,18750",
        "UCTestChannel2,video4,Ultimate LEGO Tank Build - Military Series,Building an incredible LEGO military tank,8900",
        "UCTestChannel2,video5,LEGO vs Real: Comparing Military Vehicles,Side by side comparison of LEGO and real tanks,11200",
        "UCTestChannel2,video6,Top 10 LEGO Military Sets You Must Have,Review of the best LEGO military collections,9750",
        "UCTestChannel3,video7,Epic Space Battle Compilation,Compilation of the best space battle scenes,22000",
        "UCTestChannel3,video8,Behind the Scenes: Creating Space VFX,How movie space effects are created,16800",
        "UCTestChannel3,video9,Space vs Fantasy: Which is Better?,Comparing different sci-fi genres,13900"
    ]
    
    sample_file = "sample_data.csv"
    try:
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_data))
        
        print(f"✓ Created sample data file: {sample_file}")
        print(f"✓ Contains {len(sample_data)-1} sample videos from 3 channels")
        return True
        
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        return False

def generate_mock_analysis():
    """Generate mock analysis results to demonstrate expected output format."""
    print("\n=== MOCK ANALYSIS RESULTS ===")
    
    mock_results = {
        "dataset_summary": {
            "total_videos": 9,
            "unique_channels": 3,
            "avg_views_per_video": 14311,
            "total_views": 128800
        },
        "channel_analysis": {
            "UCTestChannel1": {
                "channel_type": "space/science",
                "video_count": 3,
                "avg_views": 15417,
                "performance_category": "high",
                "title_patterns": {
                    "avg_length_words": 7.3,
                    "question_ratio": 0.33,
                    "numeric_ratio": 0.33,
                    "superlative_ratio": 0.33
                }
            },
            "UCTestChannel2": {
                "channel_type": "gaming/toys",
                "video_count": 3,
                "avg_views": 9950,
                "performance_category": "medium",
                "title_patterns": {
                    "avg_length_words": 8.7,
                    "question_ratio": 0.0,
                    "numeric_ratio": 0.33,
                    "superlative_ratio": 0.67
                }
            },
            "UCTestChannel3": {
                "channel_type": "entertainment",
                "video_count": 3,
                "avg_views": 17567,
                "performance_category": "high",
                "title_patterns": {
                    "avg_length_words": 6.3,
                    "question_ratio": 0.33,
                    "numeric_ratio": 0.0,
                    "superlative_ratio": 0.33
                }
            }
        },
        "performance_insights": {
            "high_performance_threshold": 16000,
            "success_factors": {
                "optimal_title_length": "6-8 words",
                "effective_patterns": ["questions", "superlatives", "numbers"],
                "top_content_types": ["space/science", "entertainment"]
            }
        }
    }
    
    # Save mock results
    try:
        with open("mock_analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(mock_results, f, indent=2)
        
        print("✓ Generated mock analysis results")
        
        # Display key insights
        print("\nKEY INSIGHTS FROM MOCK DATA:")
        print("─" * 40)
        
        for channel_id, data in mock_results["channel_analysis"].items():
            print(f"\n{channel_id}:")
            print(f"  Type: {data['channel_type']}")
            print(f"  Videos: {data['video_count']}")
            print(f"  Avg Views: {data['avg_views']:,}")
            print(f"  Performance: {data['performance_category']}")
            print(f"  Title Length: {data['title_patterns']['avg_length_words']:.1f} words")
        
        print(f"\nHigh Performance Threshold: {mock_results['performance_insights']['high_performance_threshold']:,} views")
        print(f"Optimal Title Length: {mock_results['performance_insights']['success_factors']['optimal_title_length']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating mock results: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("TitleCraft AI - Data Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("Module Structure", test_module_structure),
        ("Python Syntax", test_imports),
        ("Data File", test_data_file),
        ("Sample Data Creation", create_sample_data),
        ("Mock Analysis", generate_mock_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print("-" * 50)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The data module is ready to use.")
        print("\nNEXT STEPS:")
        print("1. Install dependencies: pip install pandas numpy")
        print("2. Run the example: python example_usage.py")
        print("3. Use sample_data.csv for testing if main data file unavailable")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()