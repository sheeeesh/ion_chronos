# test_imports.py
"""
Test script to verify all modules can import validation functions correctly.
"""

def test_direct_imports():
    """Test importing validation functions directly."""
    print("=== Testing Direct Imports ===")
    
    try:
        from tools.validation import verify_execution_log, get_latest_execution_summary
        print("✅ tools.validation imports successful")
    except Exception as e:
        print(f"❌ tools.validation import failed: {e}")
        return False
    
    try:
        from tools.result_checker import verify_backtest_artifacts, check_result_plausibility
        print("✅ tools.result_checker imports successful")
    except Exception as e:
        print(f"❌ tools.result_checker import failed: {e}")
        return False
    
    try:
        from tools.safe_classifier_backtest import safe_classifier_backtest
        print("✅ tools.safe_classifier_backtest imports successful")
    except Exception as e:
        print(f"❌ tools.safe_classifier_backtest import failed: {e}")
        return False
    
    return True


def test_package_imports():
    """Test importing validation functions from tools package."""
    print("\n=== Testing Package Imports ===")
    
    try:
        from tools import (
            verify_execution_log, 
            get_latest_execution_summary,
            verify_backtest_artifacts, 
            check_result_plausibility,
            safe_classifier_backtest
        )
        print("✅ All validation functions imported from tools package")
    except Exception as e:
        print(f"❌ Package import failed: {e}")
        return False
    
    return True


def test_existing_compatibility():
    """Test that existing modules still work."""
    print("\n=== Testing Existing Module Compatibility ===")
    
    try:
        from tools.classifier_backtest import classifier_backtest
        from tools.backtest import backtest_signal
        from tools.astro_dataset import build_astro_dataset
        print("✅ Existing tools modules import successfully")
    except Exception as e:
        print(f"❌ Existing module import failed: {e}")
        return False
    
    try:
        from agent.ion_chronos_agent import create_agent
        print("✅ Agent module imports successfully")
    except Exception as e:
        print(f"❌ Agent import failed: {e}")
        return False
    
    return True


def test_backend_compatibility():
    """Test that backend can import validation functions."""
    print("\n=== Testing Backend Compatibility ===")
    
    import sys
    from pathlib import Path
    
    # Add tools to path like backend does
    TOOLS = Path(r"C:\ion_chronos\tools")
    if TOOLS.exists() and str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    
    try:
        import validation as validation_mod
        import result_checker as result_checker_mod
        import safe_classifier_backtest as safe_mod
        print("✅ Backend can import validation modules")
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        return False
    
    return True


def test_functionality():
    """Test that validation functions actually work."""
    print("\n=== Testing Functionality ===")
    
    try:
        from tools.validation import get_latest_execution_summary
        result = get_latest_execution_summary("SPY")
        print(f"✅ Validation function works: {result[:50]}...")
    except Exception as e:
        print(f"❌ Validation function failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Ion Chronos Import Validation Test")
    print("=" * 50)
    
    tests = [
        test_direct_imports,
        test_package_imports, 
        test_existing_compatibility,
        test_backend_compatibility,
        test_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All import tests passed! The validation system is properly integrated.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")