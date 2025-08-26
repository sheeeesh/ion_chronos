# tools/__init__.py
"""
Ion Chronos Tools Package

This package contains all the core tools for trading analysis, backtesting,
and validation.
"""

# Core trading tools
from .astro_dataset import build_astro_dataset
from .backtest import backtest_signal
from .classifier_backtest import classifier_backtest
from .rl_train import train_rl_agent
from .pipeline import run_rl_astro_pipeline

# Validation and safety tools
from .validation import verify_execution_log, get_latest_execution_summary
from .result_checker import verify_backtest_artifacts, check_result_plausibility
from .safe_classifier_backtest import safe_classifier_backtest

# Utility tools
from .web_search import web_search
from .file_manager import write_file, read_file
from .fs_access import ls, read_text, write_text

__all__ = [
    # Core trading tools
    'build_astro_dataset',
    'backtest_signal', 
    'classifier_backtest',
    'rl_train',
    'run_rl_astro_pipeline',
    
    # Validation and safety tools
    'verify_execution_log',
    'get_latest_execution_summary',
    'verify_backtest_artifacts',
    'check_result_plausibility',
    'safe_classifier_backtest',
    
    # Utility tools
    'web_search',
    'write_file',
    'read_file',
    'ls',
    'read_text', 
    'write_text',
]