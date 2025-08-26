# tools/validation.py
"""
Validation utilities to prevent hallucinated results.
"""

import os
import json
import time
from typing import Dict, Optional
from tools.io_paths import WORKSPACE


def verify_execution_log(ticker: str, expected_threshold: Optional[float] = None, 
                        max_age_seconds: int = 300) -> Dict:
    """
    Verify that a classifier backtest was actually executed recently.
    
    Args:
        ticker: Stock symbol to check
        expected_threshold: Expected threshold value (optional)
        max_age_seconds: Maximum age of execution log in seconds (default 5 minutes)
    
    Returns:
        Dict with verification results
    """
    log_path = os.path.join(WORKSPACE, "experiments", "classifier_backtest", ticker, "execution_log.json")
    
    if not os.path.exists(log_path):
        return {
            "verified": False,
            "reason": "No execution log found",
            "log_path": log_path
        }
    
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        
        current_time = int(time.time())
        log_timestamp = log_data.get("timestamp", 0)
        age_seconds = current_time - log_timestamp
        
        if age_seconds > max_age_seconds:
            return {
                "verified": False,
                "reason": f"Execution log too old ({age_seconds}s > {max_age_seconds}s)",
                "log_data": log_data,
                "age_seconds": age_seconds
            }
        
        if expected_threshold is not None:
            actual_threshold = log_data.get("threshold")
            if abs(actual_threshold - expected_threshold) > 0.001:
                return {
                    "verified": False,
                    "reason": f"Threshold mismatch: expected {expected_threshold}, got {actual_threshold}",
                    "log_data": log_data
                }
        
        return {
            "verified": True,
            "log_data": log_data,
            "age_seconds": age_seconds
        }
        
    except Exception as e:
        return {
            "verified": False,
            "reason": f"Error reading log: {str(e)}",
            "log_path": log_path
        }


def get_latest_execution_summary(ticker: str) -> str:
    """Get a summary of the latest execution for a ticker."""
    verification = verify_execution_log(ticker)
    
    if not verification["verified"]:
        return f"❌ No recent valid execution found: {verification['reason']}"
    
    log_data = verification["log_data"]
    age_minutes = verification["age_seconds"] // 60
    
    return f"""✅ Verified execution (ID: {log_data['execution_id']})
- Age: {age_minutes} minutes ago
- Ticker: {log_data['ticker']}
- Threshold: {log_data['threshold']}
- Trades: {log_data['num_trades']}
- Return: {log_data['total_return']:.1%}"""


if __name__ == "__main__":
    # Test the validation
    print("Testing validation for SPY...")
    result = verify_execution_log("SPY")
    print(json.dumps(result, indent=2))
    
    print("\nLatest execution summary:")
    print(get_latest_execution_summary("SPY"))