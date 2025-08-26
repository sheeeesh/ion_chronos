# validation_utils.py
"""
Global validation utilities for Ion Chronos.

This module provides easy access to validation functions from anywhere in the project.
Import this module when you need validation capabilities in scripts, notebooks, or other modules.
"""

import sys
import os
from pathlib import Path

# Ensure tools directory is in path
TOOLS_DIR = Path(__file__).parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Import all validation functions
try:
    from tools.validation import verify_execution_log, get_latest_execution_summary
    from tools.result_checker import verify_backtest_artifacts, check_result_plausibility
    from tools.safe_classifier_backtest import safe_classifier_backtest
    
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import validation functions: {e}")
    VALIDATION_AVAILABLE = False


def quick_validate(ticker: str, max_age_minutes: int = 5) -> dict:
    """
    Quick validation check for a ticker's latest execution.
    
    Args:
        ticker: Stock symbol to validate
        max_age_minutes: Maximum age of execution in minutes
        
    Returns:
        Dict with validation results
    """
    if not VALIDATION_AVAILABLE:
        return {"error": "Validation functions not available"}
    
    max_age_seconds = max_age_minutes * 60
    
    # Check execution log
    execution_check = verify_execution_log(ticker, max_age_seconds=max_age_seconds)
    
    # Check artifacts if execution is valid
    artifact_check = None
    if execution_check["verified"]:
        artifact_check = verify_backtest_artifacts(ticker)
    
    return {
        "ticker": ticker,
        "execution_verified": execution_check["verified"],
        "execution_details": execution_check,
        "artifacts_verified": artifact_check["verified"] if artifact_check else False,
        "artifact_details": artifact_check,
        "summary": get_latest_execution_summary(ticker) if execution_check["verified"] else "No valid execution found"
    }


def validate_result_text(result_text: str) -> dict:
    """
    Validate result text for plausibility.
    
    Args:
        result_text: The result text to check
        
    Returns:
        Dict with plausibility analysis
    """
    if not VALIDATION_AVAILABLE:
        return {"error": "Validation functions not available"}
    
    return check_result_plausibility(result_text)


def run_safe_backtest(ticker: str, start_date: str, **kwargs) -> str:
    """
    Run a safe classifier backtest with validation.
    
    Args:
        ticker: Stock symbol
        start_date: Start date
        **kwargs: Additional arguments for classifier_backtest
        
    Returns:
        Validated backtest results
    """
    if not VALIDATION_AVAILABLE:
        return "Error: Validation functions not available"
    
    return safe_classifier_backtest(ticker=ticker, start_date=start_date, **kwargs)


# Convenience functions for common validation tasks
def is_execution_recent(ticker: str, max_age_minutes: int = 5) -> bool:
    """Check if there's a recent execution for a ticker."""
    if not VALIDATION_AVAILABLE:
        return False
    
    result = verify_execution_log(ticker, max_age_seconds=max_age_minutes * 60)
    return result["verified"]


def get_execution_age(ticker: str) -> int:
    """Get the age of the latest execution in seconds."""
    if not VALIDATION_AVAILABLE:
        return -1
    
    result = verify_execution_log(ticker, max_age_seconds=86400)  # 24 hours
    if result["verified"]:
        return result["age_seconds"]
    return -1


def list_validated_tickers() -> list:
    """List all tickers with recent valid executions."""
    if not VALIDATION_AVAILABLE:
        return []
    
    from tools.io_paths import WORKSPACE
    
    backtest_dir = Path(WORKSPACE) / "experiments" / "classifier_backtest"
    if not backtest_dir.exists():
        return []
    
    validated_tickers = []
    for ticker_dir in backtest_dir.iterdir():
        if ticker_dir.is_dir():
            ticker = ticker_dir.name
            if is_execution_recent(ticker, max_age_minutes=60):  # 1 hour
                validated_tickers.append(ticker)
    
    return validated_tickers


if __name__ == "__main__":
    print("Ion Chronos Validation Utils")
    print("=" * 40)
    
    if not VALIDATION_AVAILABLE:
        print("❌ Validation functions not available")
        sys.exit(1)
    
    print("✅ Validation functions loaded successfully")
    
    # Test with SPY if available
    print("\nTesting with SPY:")
    result = quick_validate("SPY")
    print(f"Execution verified: {result['execution_verified']}")
    print(f"Artifacts verified: {result['artifacts_verified']}")
    
    print(f"\nValidated tickers: {list_validated_tickers()}")
    
    print("\n✅ Validation utils test completed")