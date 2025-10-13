"""
Modern Pytest-based Test Runner for TitleCraft AI

This is the main test orchestrator using pytest with our modernized test suite:
- Unit Tests (Data, Processing, Services)
- Integration Tests
- Performance Tests

Usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py --unit       # Run only unit tests
    python run_all_tests.py --integration # Run only integration tests
    python run_all_tests.py --performance # Run only performance tests
    python run_all_tests.py --quick       # Run quick tests (no performance/slow)
    python run_all_tests.py --verbose     # Verbose output
    python run_all_tests.py --coverage    # Run with coverage report
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


class ModernTestRunner:
    """
    Modern pytest-based test runner for TitleCraft AI system
    
    Uses pytest with our modernized test structure for better maintainability
    """
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.total_time = 0
        self.tests_dir = Path(__file__).parent
        
    def run_pytest_suite(self, test_pattern: str, suite_name: str, additional_args: list = None):
        """Run pytest with specific pattern and arguments"""
        
        print(f"\n{'='*20} {suite_name.upper()} {'='*20}")
        
        suite_start = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.tests_dir / test_pattern),
            "-v",  # verbose output
            "--tb=short"  # shorter traceback format
        ]
        
        if additional_args:
            cmd.extend(additional_args)
        
        try:
            # Run pytest as subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            suite_time = time.time() - suite_start
            success = result.returncode == 0
            
            self.test_results[suite_name] = {
                'success': success,
                'time': suite_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr and not success:
                print("ERRORS:", result.stderr)
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status} - {suite_name} completed in {suite_time:.1f}s")
            
            return success
            
        except subprocess.TimeoutExpired:
            suite_time = time.time() - suite_start
            
            self.test_results[suite_name] = {
                'success': False,
                'time': suite_time,
                'error': "Test timeout (5 minutes exceeded)"
            }
            
            print(f"\n⏰ TIMEOUT - {suite_name} timed out after {suite_time:.1f}s")
            return False
            
        except Exception as e:
            suite_time = time.time() - suite_start
            
            self.test_results[suite_name] = {
                'success': False,
                'time': suite_time,
                'error': str(e)
            }
            
            print(f"\n💥 ERROR - {suite_name} failed with error: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        self.start_time = time.time()
        
        print("🚀 TitleCraft AI - Modern Pytest Test Suite")
        print("=" * 60)
        
        # Run all test categories
        self.run_pytest_suite("test_unit_*.py", "Unit Tests")
        self.run_pytest_suite("test_integration.py", "Integration Tests")
        self.run_pytest_suite("test_performance.py", "Performance Tests")
        
        self.total_time = time.time() - self.start_time
        self._generate_final_report()
    
    def run_unit_tests(self):
        """Run all unit tests"""
        self.start_time = time.time()
        
        print("🧪 TitleCraft AI - Unit Tests")
        print("=" * 60)
        
        self.run_pytest_suite("test_unit_*.py", "Unit Tests")
        
        self.total_time = time.time() - self.start_time
        self._generate_final_report()
    
    def run_integration_tests(self):
        """Run integration tests"""
        self.start_time = time.time()
        
        print("🔗 TitleCraft AI - Integration Tests")
        print("=" * 60)
        
        self.run_pytest_suite("test_integration.py", "Integration Tests")
        
        self.total_time = time.time() - self.start_time
        self._generate_final_report()
    
    def run_performance_tests(self):
        """Run performance tests"""
        self.start_time = time.time()
        
        print("⚡ TitleCraft AI - Performance Tests")
        print("=" * 60)
        
        self.run_pytest_suite("test_performance.py", "Performance Tests")
        
        self.total_time = time.time() - self.start_time
        self._generate_final_report()
    
    def run_quick_tests(self):
        """Run quick tests (exclude slow and performance)"""
        self.start_time = time.time()
        
        print("⚡ TitleCraft AI - Quick Test Suite")
        print("=" * 60)
        
        # Run unit and integration tests, excluding slow/performance markers
        self.run_pytest_suite(
            "test_unit_*.py test_integration.py", 
            "Quick Tests",
            ["-m", "not slow and not performance"]
        )
        
        self.total_time = time.time() - self.start_time
        self._generate_final_report()
    
    def run_with_coverage(self):
        """Run tests with coverage report"""
        self.start_time = time.time()
        
        print("📊 TitleCraft AI - Tests with Coverage")
        print("=" * 60)
        
        # Check if pytest-cov is available
        try:
            import pytest_cov
            
            self.run_pytest_suite(
                "test_unit_*.py test_integration.py",
                "Tests with Coverage",
                ["--cov=src", "--cov-report=html", "--cov-report=term-missing"]
            )
        except ImportError:
            print("⚠️  pytest-cov not installed. Running without coverage.")
            print("Install with: pip install pytest-cov")
            self.run_pytest_suite("test_unit_*.py test_integration.py", "Tests")
        
        self.total_time = time.time() - self.start_time
        self._generate_final_report()
    
    def run_verbose(self):
        """Run tests with verbose output"""
        self.start_time = time.time()
        
        print("🔍 TitleCraft AI - Verbose Test Suite")
        print("=" * 60)
        
        self.run_pytest_suite(
            "test_unit_*.py test_integration.py test_performance.py",
            "All Tests (Verbose)",
            ["-vv", "--tb=long"]
        )
        
        self.total_time = time.time() - self.start_time
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive final test report"""
        
        print("\n" + "=" * 60)
        print("📋 MODERN TEST SUITE REPORT")
        print("=" * 60)
        
        # Overall statistics
        total_suites = len(self.test_results)
        successful_suites = sum(1 for result in self.test_results.values() if result['success'])
        failed_suites = total_suites - successful_suites
        
        print(f"🔍 Test Execution Summary:")
        print(f"   • Total Test Suites: {total_suites}")
        print(f"   • ✅ Successful: {successful_suites}")
        print(f"   • ❌ Failed: {failed_suites}")
        print(f"   • ⏱️  Total Time: {self.total_time:.1f}s")
        
        if total_suites > 0:
            success_rate = (successful_suites / total_suites) * 100
            print(f"   • 🎯 Success Rate: {success_rate:.1f}%")
        
        # Detailed suite results
        print(f"\n📊 Detailed Results:")
        for suite_name, result in self.test_results.items():
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['time']:.1f}s"
            
            print(f"   {status} {suite_name:<25} ({time_str:>6})")
            
            if 'error' in result and result['error']:
                print(f"      💥 Error: {result['error']}")
            elif not result['success'] and 'stderr' in result and result['stderr']:
                # Show first line of stderr for context
                error_line = result['stderr'].split('\n')[0]
                print(f"      ⚠️  Issue: {error_line[:80]}...")
        
        # Test structure health
        print(f"\n🏗️  Test Structure Health:")
        
        unit_success = any('Unit' in name and result['success'] for name, result in self.test_results.items())
        integration_success = any('Integration' in name and result['success'] for name, result in self.test_results.items())
        performance_success = any('Performance' in name and result['success'] for name, result in self.test_results.items())
        
        if unit_success:
            print("   ✅ Unit tests are working - core components validated")
        else:
            print("   ❌ Unit tests failed - core components need attention")
        
        if integration_success:
            print("   ✅ Integration tests are working - end-to-end flow validated")
        elif 'Integration Tests' in self.test_results:
            print("   ❌ Integration tests failed - component integration issues")
        
        if performance_success:
            print("   ✅ Performance tests are working - system performance validated")
        elif 'Performance Tests' in self.test_results:
            print("   ⚠️  Performance tests failed - may need optimization")
        
        # Overall system assessment
        print(f"\n🎯 System Assessment:")
        
        if total_suites == 0:
            print("   ⚠️  No tests were run")
        elif successful_suites == total_suites:
            print("   🟢 EXCELLENT - All tests passing!")
            print("   • Test suite is comprehensive and healthy")
            print("   • System is ready for development/production")
        elif successful_suites >= total_suites * 0.8:
            print("   🟡 GOOD - Most tests passing")
            print("   • Core functionality is working")
            print("   • Some areas may need attention")
        elif successful_suites >= total_suites * 0.5:
            print("   🟠 FAIR - Significant test failures")
            print("   • Multiple components need debugging")
            print("   • Not ready for production use")
        else:
            print("   🔴 POOR - Major test failures")
            print("   • Fundamental issues with test suite or code")
            print("   • Extensive debugging required")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if successful_suites < total_suites:
            print("   1. Focus on failing tests:")
            for suite_name, result in self.test_results.items():
                if not result['success']:
                    print(f"      • Fix issues in {suite_name}")
        
        if not unit_success:
            print("   2. Unit tests are fundamental - fix these first")
            print("      • Check component imports and dependencies")
            print("      • Verify mock configurations")
        
        if unit_success and not integration_success:
            print("   2. Integration tests failing suggests component interaction issues")
            print("      • Check data flow between components")
            print("      • Verify component interfaces")
        
        print("\n🚀 Next Steps:")
        if successful_suites == total_suites:
            print("   • All tests passing - ready for development!")
            print("   • Consider adding more test coverage if needed")
        else:
            print("   • Run individual test files to isolate issues:")
            print(f"     pytest tests/test_unit_data.py -v")
            print(f"     pytest tests/test_unit_processing.py -v")
            print(f"     pytest tests/test_unit_services.py -v")
            print("   • Check component dependencies and imports")
    
    def check_pytest_setup(self):
        """Check if pytest is properly set up"""
        print("🔍 Pytest Setup Check")
        print("=" * 40)
        
        # Check if pytest is installed
        try:
            import pytest
            print(f"✅ pytest {pytest.__version__} installed")
        except ImportError:
            print("❌ pytest not installed")
            print("Install with: pip install pytest")
            return False
        
        # Check test file structure
        test_files = [
            "conftest.py",
            "pytest.ini", 
            "test_unit_data.py",
            "test_unit_processing.py",
            "test_unit_services.py",
            "test_integration.py",
            "test_performance.py"
        ]
        
        missing_files = []
        for test_file in test_files:
            if (self.tests_dir / test_file).exists():
                print(f"✅ {test_file}")
            else:
                print(f"❌ {test_file} (missing)")
                missing_files.append(test_file)
        
        # Check optional dependencies
        optional_packages = [
            ('pytest-cov', 'Coverage reporting'),
            ('pytest-xdist', 'Parallel test execution'),
            ('pytest-mock', 'Enhanced mocking capabilities')
        ]
        
        print("\nOptional pytest packages:")
        for package, description in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                print(f"✅ {package} - {description}")
            except ImportError:
                print(f"⚠️  {package} - {description} (optional)")
        
        print("=" * 40)
        
        if missing_files:
            print(f"❌ Missing {len(missing_files)} test files")
            return False
        else:
            print("✅ Pytest setup is complete!")
            return True


def main():
    """Main entry point for modern test runner"""
    
    parser = argparse.ArgumentParser(description="TitleCraft AI Modern Test Runner")
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--quick', action='store_true', help='Run quick tests (no slow/performance)')
    parser.add_argument('--verbose', action='store_true', help='Run with verbose output')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--check-setup', action='store_true', help='Check pytest setup')
    
    args = parser.parse_args()
    
    runner = ModernTestRunner()
    
    # Check setup if requested
    if args.check_setup:
        runner.check_pytest_setup()
        return
    
    # Run specific test suite based on arguments
    if args.unit:
        runner.run_unit_tests()
    elif args.integration:
        runner.run_integration_tests()
    elif args.performance:
        runner.run_performance_tests()
    elif args.quick:
        runner.run_quick_tests()
    elif args.verbose:
        runner.run_verbose()
    elif args.coverage:
        runner.run_with_coverage()
    else:
        # Run all tests by default
        runner.run_all_tests()


if __name__ == "__main__":
    main()