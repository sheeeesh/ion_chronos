"""
Classifier-gated backtest for Ion Chronos.

This tool builds a machine learning classifier to predict profitable trading opportunities
and runs a backtest using the classifier's predictions with configurable thresholds.

Features:
- Uses existing astro dataset if available, otherwise builds one
- Creates binary labels based on forward returns
- Trains a Random Forest classifier on astro + technical features
- Applies probability threshold for trade entry
- Implements take-profit and stop-loss exits
- Saves classifier model and backtest results
"""
from __future__ import annotations

import os
import json
import pickle
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Use headless backend for plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.io_paths import WORKSPACE
from tools.astro_dataset import build_astro_dataset


def _create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators and prepare features for classification."""
    df = df.copy()
    
    # Technical indicators
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['rsi'] = _calculate_rsi(df['Close'], 14)
    df['bb_upper'], df['bb_lower'] = _calculate_bollinger_bands(df['Close'], 20)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Price momentum features
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_10d'] = df['Close'].pct_change(10)
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    
    # Volume features
    df['volume_sma_10'] = df['Volume'].rolling(10).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
    
    return df


def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, lower


def _create_labels(df: pd.DataFrame, forward_days: int = 5, profit_threshold: float = 0.02) -> pd.Series:
    """Create binary labels based on forward returns."""
    forward_returns = df['Close'].pct_change(forward_days).shift(-forward_days)
    labels = (forward_returns > profit_threshold).astype(int)
    return labels


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and prepare features for the classifier."""
    # Select astro features (all columns starting with 'astro_')
    astro_cols = [col for col in df.columns if col.startswith('astro_')]
    
    # Technical features
    tech_cols = ['sma_10', 'sma_20', 'sma_50', 'rsi', 'bb_position', 
                 'return_1d', 'return_5d', 'return_10d', 'volatility_10d', 'volume_ratio']
    
    # Combine all features
    feature_cols = astro_cols + tech_cols
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    features_df = df[feature_cols].copy()
    
    # Fill NaN values with forward fill then backward fill
    features_df = features_df.ffill().bfill()
    
    return features_df


def _train_classifier(features: pd.DataFrame, labels: pd.Series, test_size: float = 0.3) -> Tuple[RandomForestClassifier, Dict]:
    """Train a Random Forest classifier and return model with metrics."""
    # Remove rows where labels are NaN
    valid_idx = ~labels.isna()
    features_clean = features[valid_idx]
    labels_clean = labels[valid_idx]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_clean, labels_clean, test_size=test_size, random_state=42, stratify=labels_clean
    )
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': clf.score(X_test, y_test),
        'feature_importance': dict(zip(features_clean.columns, clf.feature_importances_)),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return clf, metrics


def _find_optimal_threshold(clf: RandomForestClassifier, features: pd.DataFrame, labels: pd.Series) -> float:
    """Find optimal probability threshold for classification."""
    valid_idx = ~labels.isna()
    features_clean = features[valid_idx]
    labels_clean = labels[valid_idx]
    
    probabilities = clf.predict_proba(features_clean)[:, 1]
    
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.arange(0.5, 0.95, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        if predictions.sum() > 0:  # Ensure we have some positive predictions
            precision = (predictions & labels_clean).sum() / predictions.sum()
            recall = (predictions & labels_clean).sum() / labels_clean.sum()
            if precision > 0 and recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_score:
                    best_score = f1
                    best_threshold = threshold
    
    return best_threshold


def _run_classifier_backtest(df: pd.DataFrame, clf: RandomForestClassifier, features: pd.DataFrame,
                           threshold: float = 0.7, take_profit: float = 0.005, stop_loss: float = 0.01,
                           cost: float = 0.001) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    """Run backtest using classifier predictions."""
    # Get predictions
    probabilities = clf.predict_proba(features)[:, 1]
    signals = pd.Series(probabilities >= threshold, index=df.index)
    
    # Initialize tracking variables
    equity = pd.Series(1.0, index=df.index)
    trades = []
    in_position = False
    entry_price = 0
    entry_time = None
    
    for i, (date, row) in enumerate(df.iterrows()):
        if i == 0:
            continue
            
        current_price = row['Close']
        signal = signals.iloc[i]
        
        # Check for entry
        if not in_position and signal:
            in_position = True
            entry_price = current_price
            entry_time = date
            equity.iloc[i] = equity.iloc[i-1] * (1 - cost)  # Entry cost
            
        # Check for exit
        elif in_position:
            price_change = (current_price - entry_price) / entry_price
            
            # Take profit or stop loss
            if price_change >= take_profit or price_change <= -stop_loss:
                # Exit position
                gross_return = price_change
                net_return = gross_return - 2 * cost  # Entry + exit costs
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'gross_return': gross_return,
                    'net_return': net_return,
                    'exit_reason': 'take_profit' if price_change >= take_profit else 'stop_loss'
                })
                
                equity.iloc[i] = equity.iloc[i-1] * (1 + net_return)
                in_position = False
            else:
                # Hold position
                equity.iloc[i] = equity.iloc[i-1] * (1 + price_change - cost/252)  # Daily holding cost
        else:
            # No position, no change
            equity.iloc[i] = equity.iloc[i-1]
    
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    total_return = equity.iloc[-1] - 1
    daily_returns = equity.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
    max_dd = (equity / equity.cummax() - 1).min()
    
    metrics = {
        'total_return': total_return,
        'annualized_return': (equity.iloc[-1] ** (252 / len(equity))) - 1,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'num_trades': len(trades_df),
        'win_rate': (trades_df['net_return'] > 0).mean() if len(trades_df) > 0 else 0,
        'avg_return_per_trade': trades_df['net_return'].mean() if len(trades_df) > 0 else 0
    }
    
    return equity, trades_df, metrics


def _save_artifacts(ticker: str, clf: RandomForestClassifier, threshold: float, metrics: Dict,
                   equity: pd.Series, trades_df: pd.DataFrame, backtest_metrics: Dict) -> List[str]:
    """Save all artifacts and return list of file paths."""
    artifacts = []
    
    # Create directories
    classifier_dir = os.path.join(WORKSPACE, "experiments", "classifier")
    backtest_dir = os.path.join(WORKSPACE, "experiments", "classifier_backtest", ticker)
    os.makedirs(classifier_dir, exist_ok=True)
    os.makedirs(backtest_dir, exist_ok=True)
    
    # Add timestamp to all saved files for validation
    import time
    timestamp = int(time.time())
    print(f"[VALIDATION] Saving artifacts with timestamp {timestamp}")
    
    # Save execution log for validation
    log_path = os.path.join(backtest_dir, "execution_log.json")
    execution_log = {
        "timestamp": timestamp,
        "ticker": ticker,
        "threshold": threshold,
        "num_trades": backtest_metrics['num_trades'],
        "total_return": backtest_metrics['total_return'],
        "execution_id": f"EXEC_{timestamp}"
    }
    with open(log_path, 'w') as f:
        json.dump(execution_log, f, indent=2)
    artifacts.append(log_path)
    
    # Save classifier model
    model_path = os.path.join(classifier_dir, f"{ticker}_rf.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    artifacts.append(model_path)
    
    # Save threshold
    threshold_path = os.path.join(classifier_dir, f"{ticker}_threshold.json")
    with open(threshold_path, 'w') as f:
        json.dump({'threshold': threshold, 'metrics': metrics}, f, indent=2)
    artifacts.append(threshold_path)
    
    # Save trades
    trades_path = os.path.join(backtest_dir, "trades.csv")
    trades_df.to_csv(trades_path, index=False)
    artifacts.append(trades_path)
    
    # Save summary
    summary_path = os.path.join(backtest_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(backtest_metrics, f, indent=2)
    artifacts.append(summary_path)
    
    # Create equity plot
    equity_path = os.path.join(backtest_dir, "equity.png")
    plt.figure(figsize=(12, 6))
    plt.plot(equity.index, equity.values)
    plt.title(f'{ticker} Classifier-Gated Backtest - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(equity_path, dpi=150, bbox_inches='tight')
    plt.close()
    artifacts.append(equity_path)
    
    # Create drawdown plot
    drawdown = equity / equity.cummax() - 1
    drawdown_path = os.path.join(backtest_dir, "drawdown.png")
    plt.figure(figsize=(12, 6))
    plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    plt.plot(drawdown.index, drawdown.values, color='red')
    plt.title(f'{ticker} Classifier-Gated Backtest - Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(drawdown_path, dpi=150, bbox_inches='tight')
    plt.close()
    artifacts.append(drawdown_path)
    
    return artifacts


def classifier_backtest(ticker: str,
                       start_date: str,
                       end_date: Optional[str] = None,
                       threshold: Optional[float] = None,
                       take_profit: float = 0.005,
                       stop_loss: float = 0.01,
                       cost: float = 0.001,
                       forward_days: int = 5,
                       profit_threshold: float = 0.02) -> str:
    """
    Run a classifier-gated backtest.
    
    Args:
        ticker: Stock symbol (e.g., 'SPY')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format (optional)
        threshold: Probability threshold for trade entry (auto-optimized if None)
        take_profit: Take profit level as fraction (e.g., 0.005 = 0.5%)
        stop_loss: Stop loss level as fraction (e.g., 0.01 = 1%)
        cost: Transaction cost per trade as fraction
        forward_days: Days to look forward for labeling
        profit_threshold: Minimum return for positive label
    
    Returns:
        Summary string with results and artifact locations
    """
    
    # Add execution timestamp for validation
    import time
    execution_id = f"EXEC_{int(time.time())}"
    print(f"[VALIDATION] Starting classifier_backtest execution {execution_id}")
    
    # Step 1: Load or build astro dataset
    dataset_path = os.path.join(WORKSPACE, f"{ticker}_astro_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"Building astro dataset for {ticker}...")
        build_astro_dataset(ticker, start_date, end_date)
    
    # Load data
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    
    # Filter date range
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    print(f"Loaded {len(df)} rows of data for {ticker}")
    
    # Step 2: Create features and labels
    df = _create_features(df)
    labels = _create_labels(df, forward_days, profit_threshold)
    features = _prepare_features(df)
    
    print(f"Created {len(features.columns)} features")
    print(f"Positive labels: {labels.sum()}/{len(labels)} ({labels.mean():.1%})")
    
    # Step 3: Train classifier
    print("Training classifier...")
    clf, train_metrics = _train_classifier(features, labels)
    
    # Step 4: Find optimal threshold if not provided
    if threshold is None:
        threshold = _find_optimal_threshold(clf, features, labels)
        print(f"Optimal threshold: {threshold:.3f}")
    
    # Step 5: Run backtest
    print("Running backtest...")
    equity, trades_df, backtest_metrics = _run_classifier_backtest(
        df, clf, features, threshold, take_profit, stop_loss, cost
    )
    
    # Step 6: Save artifacts
    artifacts = _save_artifacts(ticker, clf, threshold, train_metrics, equity, trades_df, backtest_metrics)
    
    # Step 7: Create summary with validation markers
    summary = f"""
[VALIDATION] Classifier_backtest execution {execution_id} COMPLETED

Classifier-Gated Backtest Results for {ticker}

Dataset: {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}
Features: {len(features.columns)} (astro + technical)
Threshold: {threshold:.3f}

Classifier Performance:
- Accuracy: {train_metrics['accuracy']:.3f}
- Positive Labels: {labels.sum()}/{len(labels)} ({labels.mean():.1%})

Backtest Results:
- Total Return: {backtest_metrics['total_return']:.1%}
- Annualized Return: {backtest_metrics['annualized_return']:.1%}
- Volatility: {backtest_metrics['volatility']:.1%}
- Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}
- Max Drawdown: {backtest_metrics['max_drawdown']:.1%}
- Number of Trades: {backtest_metrics['num_trades']}
- Win Rate: {backtest_metrics['win_rate']:.1%}
- Avg Return per Trade: {backtest_metrics['avg_return_per_trade']:.2%}

Artifacts saved:
""" + "\n".join(f"- {path}" for path in artifacts) + f"""

[VALIDATION] Execution {execution_id} timestamp: {int(time.time())}
"""
    
    print(f"[VALIDATION] Completed classifier_backtest execution {execution_id}")
    return summary


if __name__ == "__main__":
    # Test with SPY
    result = classifier_backtest("SPY", "2020-01-01", "2024-12-31")
    print(result)