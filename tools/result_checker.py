# tools/result_checker.py
"""
Result verification utilities to catch hallucinated outputs.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional
from tools.io_paths import WORKSPACE


def verify_backtest_artifacts(ticker: str, expected_trades: Optional[int] = None) -> Dict:
    """
    Verify that backtest artifacts exist and contain expected data.
    
    Args:
        ticker: Stock symbol
        expected_trades: Expected number of trades (optional)
    
    Returns:
        Dict with verification results
    """
    backtest_dir = os.path.join(WORKSPACE, "experiments", "classifier_backtest", ticker)
    
    required_files = [
        "summary.json",
        "trades.csv", 
        "equity.png",
        "drawdown.png",
        "execution_log.json"
    ]
    
    verification = {
        "verified": True,
        "missing_files": [],
        "file_checks": {},
        "data_checks": {}
    }
    
    # Check file existence
    for filename in required_files:
        filepath = os.path.join(backtest_dir, filename)
        exists = os.path.exists(filepath)
        verification["file_checks"][filename] = exists
        
        if not exists:
            verification["verified"] = False
            verification["missing_files"].append(filename)
    
    if not verification["verified"]:
        return verification
    
    # Check file contents
    try:
        # Verify summary.json
        summary_path = os.path.join(backtest_dir, "summary.json")
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
        
        actual_trades = summary_data.get("num_trades", 0)
        verification["data_checks"]["actual_trades"] = actual_trades
        
        if expected_trades is not None and actual_trades != expected_trades:
            verification["verified"] = False
            verification["data_checks"]["trade_mismatch"] = f"Expected {expected_trades}, got {actual_trades}"
        
        # Verify trades.csv has correct number of rows
        trades_path = os.path.join(backtest_dir, "trades.csv")
        trades_df = pd.read_csv(trades_path)
        csv_trades = len(trades_df)
        verification["data_checks"]["csv_trades"] = csv_trades
        
        if csv_trades != actual_trades:
            verification["verified"] = False
            verification["data_checks"]["csv_mismatch"] = f"Summary says {actual_trades} trades, CSV has {csv_trades}"
        
        # Check for reasonable values
        total_return = summary_data.get("total_return", 0)
        if abs(total_return) > 10:  # More than 1000% return is suspicious
            verification["verified"] = False
            verification["data_checks"]["suspicious_return"] = f"Return of {total_return:.1%} seems unrealistic"
        
        verification["data_checks"]["summary_data"] = summary_data
        
    except Exception as e:
        verification["verified"] = False
        verification["data_checks"]["error"] = str(e)
    
    return verification


def check_result_plausibility(result_text: str) -> Dict:
    """
    Check if reported results seem plausible based on common patterns.
    
    Args:
        result_text: The result text to analyze
    
    Returns:
        Dict with plausibility analysis
    """
    import re
    
    analysis = {
        "plausible": True,
        "warnings": [],
        "extracted_metrics": {}
    }
    
    # Extract metrics using regex
    patterns = {
        "trades": r"Number of Trades:\s*(\d+)",
        "win_rate": r"Win Rate:\s*([\d.]+)%",
        "total_return": r"Total Return:\s*([\d.-]+)%",
        "sharpe": r"Sharpe Ratio:\s*([\d.-]+)"
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, result_text)
        if match:
            try:
                value = float(match.group(1))
                analysis["extracted_metrics"][metric] = value
            except ValueError:
                pass
    
    # Check for suspicious patterns
    metrics = analysis["extracted_metrics"]
    
    # Suspiciously high win rates
    if "win_rate" in metrics and metrics["win_rate"] > 95:
        analysis["warnings"].append(f"Win rate of {metrics['win_rate']:.1f}% is unusually high")
    
    # Suspiciously high returns
    if "total_return" in metrics and abs(metrics["total_return"]) > 100:
        analysis["warnings"].append(f"Return of {metrics['total_return']:.1f}% is unusually high")
    
    # Too many trades for short periods
    if "trades" in metrics and metrics["trades"] > 200:
        analysis["warnings"].append(f"{metrics['trades']} trades seems excessive for typical backtests")
    
    # Perfect Sharpe ratios
    if "sharpe" in metrics and metrics["sharpe"] > 5:
        analysis["warnings"].append(f"Sharpe ratio of {metrics['sharpe']:.2f} is unusually high")
    
    # Check for validation markers
    if "[VALIDATION]" not in result_text:
        analysis["warnings"].append("No validation markers found - execution may not be real")
    
    if analysis["warnings"]:
        analysis["plausible"] = False
    
    return analysis


if __name__ == "__main__":
    # Test the verification
    print("Testing artifact verification for SPY...")
    result = verify_backtest_artifacts("SPY")
    print(json.dumps(result, indent=2, default=str))
    
    # Test plausibility check
    test_result = """
    Total Return: 9.8%
    Number of Trades: 16
    Win Rate: 87.5%
    Sharpe Ratio: 1.89
    [VALIDATION] Execution EXEC_123 COMPLETED
    """
    
    print("\nTesting plausibility check...")
    plausibility = check_result_plausibility(test_result)
    print(json.dumps(plausibility, indent=2))