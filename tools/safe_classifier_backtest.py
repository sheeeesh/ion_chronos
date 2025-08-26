# tools/safe_classifier_backtest.py
"""
Safe wrapper for classifier_backtest with automatic validation.
"""

from typing import Optional
from tools.classifier_backtest import classifier_backtest
from tools.validation import verify_execution_log
from tools.result_checker import verify_backtest_artifacts, check_result_plausibility
import json


def safe_classifier_backtest(ticker: str,
                            start_date: str,
                            end_date: Optional[str] = None,
                            threshold: Optional[float] = None,
                            take_profit: float = 0.005,
                            stop_loss: float = 0.01,
                            cost: float = 0.001,
                            forward_days: int = 5,
                            profit_threshold: float = 0.02) -> str:
    """
    Run classifier_backtest with automatic validation to prevent hallucinated results.
    
    This wrapper:
    1. Runs the actual classifier_backtest
    2. Verifies execution logs
    3. Checks artifact integrity
    4. Validates result plausibility
    5. Returns enhanced results with validation status
    """
    
    print(f"[SAFE_BACKTEST] Starting validated classifier backtest for {ticker}")
    
    # Step 1: Run the actual backtest
    try:
        result = classifier_backtest(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            threshold=threshold,
            take_profit=take_profit,
            stop_loss=stop_loss,
            cost=cost,
            forward_days=forward_days,
            profit_threshold=profit_threshold
        )
    except Exception as e:
        return f"❌ BACKTEST FAILED: {str(e)}"
    
    # Step 2: Verify execution log
    execution_verification = verify_execution_log(ticker, threshold, max_age_seconds=60)
    if not execution_verification["verified"]:
        return f"❌ EXECUTION VERIFICATION FAILED: {execution_verification['reason']}\n\nOriginal result:\n{result}"
    
    # Step 3: Verify artifacts
    artifact_verification = verify_backtest_artifacts(ticker)
    if not artifact_verification["verified"]:
        missing = ", ".join(artifact_verification["missing_files"])
        return f"❌ ARTIFACT VERIFICATION FAILED: Missing files: {missing}\n\nOriginal result:\n{result}"
    
    # Step 4: Check plausibility
    plausibility_check = check_result_plausibility(result)
    if not plausibility_check["plausible"]:
        warnings = "; ".join(plausibility_check["warnings"])
        return f"⚠️ PLAUSIBILITY WARNING: {warnings}\n\nOriginal result:\n{result}"
    
    # Step 5: Return validated result
    validation_summary = f"""
✅ VALIDATION PASSED - Results are verified as authentic

Execution ID: {execution_verification['log_data']['execution_id']}
Artifacts verified: {len(artifact_verification['file_checks'])} files
Data integrity: ✅ Summary matches CSV trades
Plausibility: ✅ All metrics within reasonable ranges

{result}
"""
    
    print(f"[SAFE_BACKTEST] Validation completed successfully for {ticker}")
    return validation_summary


if __name__ == "__main__":
    # Test the safe wrapper
    print("Testing safe classifier backtest...")
    result = safe_classifier_backtest(
        ticker="SPY",
        start_date="2023-01-01", 
        end_date="2023-12-31",
        threshold=0.3,
        forward_days=1,
        profit_threshold=0.005
    )
    print(result)