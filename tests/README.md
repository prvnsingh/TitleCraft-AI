# TitleCraft AI - Test Suite

This directory contains a comprehensive and consistent test suite for the TitleCraft AI project.

## Test Files Overview

### Main Test Suites

1. **`test_quick.py`** - ⚡ Recommended for regular use
   - Fast validation of core functionality
   - Simple pass/fail reporting
   - Tests both offline and live scenarios
   - **Best for:** Quick validation during development

2. **`test_comprehensive.py`** - 🧪 Full unittest suite
   - Organized test classes using Python unittest
   - Detailed test isolation and setup/teardown
   - Comprehensive coverage of all components
   - **Best for:** Thorough testing and CI/CD

3. **`validate_assignment.py`** - 📋 Assignment validation
   - Validates all assignment requirements
   - Structured requirement checking
   - Comprehensive project validation
   - **Best for:** Final assignment validation

### Specific Test Files

4. **`test_direct.py`** - 🔧 Direct functionality testing
   - Tests core functionality without API server
   - No network dependencies
   - **Best for:** Testing without running server

5. **`simple_test.py`** - 🌐 Simple API testing
   - Basic API endpoint validation
   - Requires running server
   - **Best for:** Quick API health checks

6. **`test_api.py`** - 🔍 Detailed API testing
   - Comprehensive API endpoint testing
   - Error handling validation
   - **Best for:** Thorough API validation

### Test Runner

7. **`run_tests.py`** - 🚀 Test runner script
   - Unified interface for all tests
   - Multiple test suite options
   - Summary reporting

## Usage

### Quick Testing (Recommended)
```bash
python tests/test_quick.py
# OR
python tests/run_tests.py quick
```

### Run All Tests
```bash
python tests/run_tests.py all
```

### Specific Test Types
```bash
python tests/run_tests.py direct        # Direct functionality only
python tests/run_tests.py simple        # Simple API tests  
python tests/run_tests.py api          # Detailed API tests
python tests/run_tests.py comprehensive # Full unittest suite
python tests/run_tests.py validate     # Assignment validation
```

### Individual Test Files
```bash
python tests/test_direct.py             # Direct functionality
python tests/simple_test.py             # Simple API tests (requires server)
python tests/test_api.py               # Detailed API tests (requires server)
python tests/validate_assignment.py    # Assignment validation
```

## Test Categories

### 🔧 Core Functionality Tests
- Data loading from CSV
- Channel analysis and pattern detection
- Title generation (fallback mode)
- API model structure

### 🌐 API Tests  
- Health endpoint (`/health`)
- Root endpoint (`/`)
- Title generation endpoint (`/generate`)
- Error handling for invalid inputs

### 🛡️ Error Handling Tests
- Missing OpenAI API key handling
- Invalid channel ID handling
- Malformed request handling
- Graceful degradation

### 📊 Data Tests
- CSV data loading (211 videos)
- Channel data retrieval
- Pattern analysis
- Performance metrics calculation

## Test Consistency Features

All test files now use:
- ✅ Consistent channel ID for testing: `UC510QYlOlKNyhy_zdQxnGYw`
- ✅ Standardized output formatting with emojis
- ✅ Proper error handling and reporting
- ✅ Consistent test data and scenarios
- ✅ Clear pass/fail indicators
- ✅ Proper cleanup and teardown

## Server Requirements

### Tests that need a running server:
- `simple_test.py`
- `test_api.py` (live tests)
- `run_tests.py all` (includes live tests)

Start server: `python -m src.apis.app`

### Tests that work without server:
- `test_quick.py` (has both offline and optional live tests)
- `test_direct.py`
- `test_comprehensive.py` (includes offline tests)
- `validate_assignment.py`

## Test Results Interpretation

### ✅ Success Indicators
- All core functionality working
- API endpoints responding correctly
- Data loading and processing successful
- Error handling working properly

### ❌ Failure Indicators  
- Missing dependencies
- Server not running (for API tests)
- Data file issues
- Import or configuration problems

### ⏭️ Skip Indicators
- Server not available (gracefully handled)
- Optional components missing
- Environment-specific features

## Recommendations

1. **During Development:** Use `python tests/test_quick.py`
2. **Before Commits:** Use `python tests/run_tests.py all`  
3. **CI/CD Pipeline:** Use `python tests/test_comprehensive.py`
4. **Assignment Submission:** Use `python tests/validate_assignment.py`
5. **Debugging Issues:** Use individual test files as needed

The test suite is designed to be robust, consistent, and provide clear feedback on the system's health and functionality.