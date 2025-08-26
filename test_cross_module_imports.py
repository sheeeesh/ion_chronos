# test_cross_module_imports.py
"""
Test that all pre-existing modules can properly import and use the new validation functions.
"""

import sys
import os
from pathlib import Path

# Ensure we're in the right directory
os.chdir(r"c:\ion_chronos")
sys.path.insert(0, r"c:\ion_chronos")

def test_tools_package_access():
    """Test that validation functions are accessible through tools package."""
    print("=== Testing Tools Package Access ===")
    
    try:
        # Test importing through tools package
        from tools import (
            verify_execution_log,
            get_latest_execution_summary, 
            verify_backtest_artifacts,
            check_result_plausibility,
            safe_classifier_backtest
        )
        print("✅ All validation functions accessible through tools package")
        return True
    except Exception as e:
        print(f"❌ Tools package access failed: {e}")
        return False


def test_direct_module_access():
    """Test direct module imports."""
    print("\n=== Testing Direct Module Access ===")
    
    modules_to_test = [
        ("tools.validation", ["verify_execution_log", "get_latest_execution_summary"]),
        ("tools.result_checker", ["verify_backtest_artifacts", "check_result_plausibility"]),
        ("tools.safe_classifier_backtest", ["safe_classifier_backtest"]),
    ]
    
    all_passed = True
    for module_name, functions in modules_to_test:
        try:
            module = __import__(module_name, fromlist=functions)
            for func_name in functions:
                if hasattr(module, func_name):
                    print(f"✅ {module_name}.{func_name}")
                else:
                    print(f"❌ {module_name}.{func_name} not found")
                    all_passed = False
        except Exception as e:
            print(f"❌ {module_name} import failed: {e}")
            all_passed = False
    
    return all_passed


def test_backend_style_imports():
    """Test imports the way backend modules do it."""
    print("\n=== Testing Backend-Style Imports ===")
    
    # Simulate backend import style
    TOOLS = Path(r"C:\ion_chronos\tools")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    
    try:
        import validation as validation_mod
        import result_checker as result_checker_mod
        import safe_classifier_backtest as safe_mod
        
        # Test that functions are accessible
        assert hasattr(validation_mod, 'verify_execution_log')
        assert hasattr(result_checker_mod, 'verify_backtest_artifacts')
        assert hasattr(safe_mod, 'safe_classifier_backtest')
        
        print("✅ Backend-style imports work correctly")
        return True
    except Exception as e:
        print(f"❌ Backend-style imports failed: {e}")
        return False


def test_validation_utils_access():
    """Test global validation utils access."""
    print("\n=== Testing Validation Utils Access ===")
    
    try:
        from validation_utils import (
            quick_validate,
            validate_result_text,
            run_safe_backtest,
            is_execution_recent,
            get_execution_age,
            list_validated_tickers
        )
        print("✅ All validation utils functions accessible")
        return True
    except Exception as e:
        print(f"❌ Validation utils access failed: {e}")
        return False


def test_agent_integration():
    """Test that agent module can access validation functions."""
    print("\n=== Testing Agent Integration ===")
    
    try:
        # Test that agent module imports successfully (it imports validation functions internally)
        import agent.ion_chronos_agent
        
        # Check that the agent module has the validation imports in its source
        import inspect
        source = inspect.getsource(agent.ion_chronos_agent)
        
        required_imports = [
            'from tools.validation import',
            'from tools.result_checker import',
            'from tools.safe_classifier_backtest import'
        ]
        
        missing_imports = []
        for imp in required_imports:
            if imp not in source:
                missing_imports.append(imp)
        
        if missing_imports:
            print(f"❌ Agent missing imports: {missing_imports}")
            return False
        
        print("✅ Agent module has all validation imports")
        return True
    except Exception as e:
        print(f"❌ Agent integration test failed: {e}")
        return False


def test_cross_module_functionality():
    """Test that modules can actually use validation functions."""
    print("\n=== Testing Cross-Module Functionality ===")
    
    try:
        # Test that we can call validation functions from different import styles
        from tools import verify_execution_log as tools_verify
        from tools.validation import verify_execution_log as direct_verify
        from validation_utils import quick_validate
        
        # These should all work (even if they return "no execution found")
        result1 = tools_verify("TEST", max_age_seconds=60)
        result2 = direct_verify("TEST", max_age_seconds=60)
        result3 = quick_validate("TEST", max_age_minutes=1)
        
        print("✅ Cross-module functionality works")
        return True
    except Exception as e:
        print(f"❌ Cross-module functionality failed: {e}")
        return False


def test_workspace_backend_compatibility():
    """Test that workspace backend can use validation functions."""
    print("\n=== Testing Workspace Backend Compatibility ===")
    
    try:
        # Simulate workspace backend environment
        workspace_backend_path = Path(r"c:\ion_chronos\workspace\backend")
        if str(workspace_backend_path) not in sys.path:
            sys.path.insert(0, str(workspace_backend_path))
        
        # Import job_runner to test its updated imports
        from workspace.backend.job_runner import run_job
        
        # Test that validation modules are available in backend context
        import validation as val_mod
        import result_checker as check_mod
        
        print("✅ Workspace backend can access validation functions")
        return True
    except Exception as e:
        print(f"❌ Workspace backend compatibility failed: {e}")
        return False


def main():
    """Run all cross-module import tests."""
    print("Ion Chronos Cross-Module Import Test")
    print("=" * 50)
    
    tests = [
        test_tools_package_access,
        test_direct_module_access,
        test_backend_style_imports,
        test_validation_utils_access,
        test_agent_integration,
        test_cross_module_functionality,
        test_workspace_backend_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ All pre-existing modules can access validation functions")
        print("✅ Import compatibility is complete")
    else:
        print("⚠️ Some tests failed - check output above")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)