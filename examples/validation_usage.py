# examples/validation_usage.py
"""
Example showing how to use the new validation functions in your own modules.
"""

# Method 1: Import directly from specific modules
from tools.validation import verify_execution_log, get_latest_execution_summary
from tools.result_checker import verify_backtest_artifacts, check_result_plausibility
from tools.safe_classifier_backtest import safe_classifier_backtest

# Method 2: Import from tools package (recommended)
from tools import (
    verify_execution_log,
    verify_backtest_artifacts, 
    safe_classifier_backtest
)

def example_safe_backtest():
    """Example of running a safe backtest with validation."""
    print("Running safe classifier backtest...")
    
    result = safe_classifier_backtest(
        ticker="SPY",
        start_date="2023-01-01",
        end_date="2023-12-31", 
        threshold=0.3,
        forward_days=1,
        profit_threshold=0.005
    )
    
    print("Backtest completed with validation:")
    print(result[:200] + "...")
    
    return result


def example_manual_validation():
    """Example of manually validating a backtest execution."""
    ticker = "SPY"
    
    print(f"Checking execution for {ticker}...")
    
    # Check execution log
    execution_check = verify_execution_log(ticker, max_age_seconds=300)
    if execution_check["verified"]:
        print("✅ Execution verified")
        print(f"   - Execution ID: {execution_check['log_data']['execution_id']}")
        print(f"   - Age: {execution_check['age_seconds']} seconds")
    else:
        print(f"❌ Execution verification failed: {execution_check['reason']}")
        return
    
    # Check artifacts
    artifact_check = verify_backtest_artifacts(ticker)
    if artifact_check["verified"]:
        print("✅ Artifacts verified")
        print(f"   - Files checked: {len(artifact_check['file_checks'])}")
        print(f"   - Trades: {artifact_check['data_checks']['actual_trades']}")
    else:
        print(f"❌ Artifact verification failed")
        print(f"   - Missing files: {artifact_check['missing_files']}")
    
    # Get summary
    summary = get_latest_execution_summary(ticker)
    print(f"Summary: {summary}")


def example_result_validation():
    """Example of validating result text for plausibility."""
    
    # Simulate some result text
    suspicious_result = """
    Total Return: 150%
    Number of Trades: 500
    Win Rate: 98.5%
    Sharpe Ratio: 8.2
    """
    
    realistic_result = """
    [VALIDATION] Execution EXEC_123 COMPLETED
    Total Return: 12.5%
    Number of Trades: 25
    Win Rate: 68.0%
    Sharpe Ratio: 1.4
    """
    
    print("Checking suspicious result:")
    check1 = check_result_plausibility(suspicious_result)
    print(f"Plausible: {check1['plausible']}")
    if check1['warnings']:
        print(f"Warnings: {'; '.join(check1['warnings'])}")
    
    print("\nChecking realistic result:")
    check2 = check_result_plausibility(realistic_result)
    print(f"Plausible: {check2['plausible']}")
    if check2['warnings']:
        print(f"Warnings: {'; '.join(check2['warnings'])}")
    else:
        print("No warnings - result looks good!")


if __name__ == "__main__":
    print("Ion Chronos Validation Usage Examples")
    print("=" * 50)
    
    print("\n1. Manual Validation Example:")
    example_manual_validation()
    
    print("\n2. Result Plausibility Check Example:")
    example_result_validation()
    
    print("\n3. Safe Backtest Example:")
    # Uncomment to run actual backtest
    # example_safe_backtest()
    print("(Safe backtest example available - uncomment to run)")
    
    print("\n✅ All examples completed!")