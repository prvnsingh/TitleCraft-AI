#!/usr/bin/env python3
"""
Test runner for TitleCraft AI
Allows running different test suites based on needs
"""
import sys
import os
import argparse
import subprocess

def run_test_file(test_file: str, description: str):
    """Run a specific test file"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join("tests", test_file)
        ], capture_output=False, text=True)
        
        return result.returncode == 0
    
    except Exception as e:
        print(f"❌ Failed to run {test_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='TitleCraft AI Test Runner')
    parser.add_argument('test_type', nargs='?', default='quick',
                       choices=['quick', 'comprehensive', 'simple', 'direct', 'api', 'validate', 'all'],
                       help='Type of test to run')
    
    args = parser.parse_args()
    
    print("TitleCraft AI - Test Runner")
    print("=" * 40)
    
    test_configs = {
        'quick': ('test_quick.py', 'Quick validation test suite'),
        'comprehensive': ('test_comprehensive.py', 'Full unittest-based test suite'),
        'simple': ('simple_test.py', 'Simple API endpoint tests'),
        'direct': ('test_direct.py', 'Direct functionality tests (no server)'),
        'api': ('test_api.py', 'Detailed API endpoint tests'),
        'validate': ('validate_assignment.py', 'Assignment requirement validation')
    }
    
    if args.test_type == 'all':
        print("Running all test suites...\n")
        results = []
        
        for test_key in ['direct', 'simple', 'api', 'quick', 'validate']:
            if test_key in test_configs:
                test_file, description = test_configs[test_key]
                success = run_test_file(test_file, description)
                results.append((description, success))
        
        # Summary
        print(f"\n{'='*60}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for description, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} - {description}")
        
        print(f"\nOverall: {passed}/{total} test suites passed")
        
        if passed == total:
            print("🎉 ALL TEST SUITES PASSED!")
        else:
            print(f"⚠️  {total - passed} test suite(s) failed")
        
        return passed == total
    
    else:
        if args.test_type in test_configs:
            test_file, description = test_configs[args.test_type]
            return run_test_file(test_file, description)
        else:
            print(f"❌ Unknown test type: {args.test_type}")
            return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)