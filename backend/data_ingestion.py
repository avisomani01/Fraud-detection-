# backend/data_ingestion.py
from __future__ import annotations
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Iterable, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# Phase 1: Data Ingestion
# -----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV and provide quick inspection."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(3).to_string())
    return df

# Add Synthetic Features
def add_synthetic_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add merchant, payment, location, and user features to increase realism.
    Use deterministic RNG with seed to keep reproducibility.
    """
    rng = np.random.RandomState(seed)

    merchant_types = ['grocery', 'electronics', 'travel', 'fashion', 'utilities', 'restaurants']
    payment_methods = ['credit_card', 'debit_card', 'digital_wallet', 'bank_transfer']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'San Francisco']

    df = df.copy()
    n = len(df)

    df['merchant_type'] = rng.choice(merchant_types, size=n)
    df['payment_method'] = rng.choice(payment_methods, size=n)
    df['location'] = rng.choice(locations, size=n)
    df['user_id'] = rng.randint(1000, 2000, size=n)

    # Convert "Time" or other seconds-since-start column into timestamps (fallback to index)
    if 'Time' in df.columns:
        df['timestamp'] = pd.to_timedelta(df['Time'], unit='s') + pd.Timestamp('2023-01-01')
    else:
        df['timestamp'] = pd.Timestamp('2023-01-01') + pd.to_timedelta(df.index, unit='s')

    # Unique transaction ID
    df['transaction_id'] = np.arange(1, n + 1)

    print('[INFO] Synthetic features added successfully.')
    return df

# -----------------------------
# Phase 2: Preprocessing & Feature Engineering
# -----------------------------
def preprocess_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None
                       ) -> Tuple[pd.DataFrame, StandardScaler]:
    """Preprocess dataset and generate features.
    Returns processed df and the fitted scaler (so downstream pipelines can use the same scaler).
    """
    df = df.copy()

    # 1. Basic data cleaning
    df.fillna(0, inplace=True)

    # 2. Encode categorical features (one-hot)
    categorical_cols = [c for c in ['merchant_type', 'payment_method', 'location'] if c in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 3. Time-based / rolling features
    if 'timestamp' in df.columns and 'user_id' in df.columns:
        # ensure timestamp dtype
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp'])
        df['time_since_last_txn'] = (
            df.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        )

        # rolling count per user in last hour example
        try:
            df['txns_last_hour'] = (
                df.set_index('timestamp')
                  .groupby('user_id')['transaction_id']
                  .rolling('1h')
                  .count()
                  .reset_index(level=0, drop=True)
                  .fillna(0)
            )
            df.reset_index(drop=True, inplace=True)
        except Exception as e:
            print(f"[WARN] rolling 'txns_last_hour' failed: {e}")
            df['txns_last_hour'] = 0

    # 4. Scale numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude target-like and ids
    for excl in ('Class', 'transaction_id'):
        if excl in numeric_cols:
            numeric_cols.remove(excl)
    if scaler is None:
        scaler = StandardScaler()
        if numeric_cols:
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        if numeric_cols:
            df[numeric_cols] = scaler.transform(df[numeric_cols])

    print('[INFO] Preprocessing and feature engineering completed.')
    return df, scaler

# -----------------------------
# Advanced Feature Engineering
# -----------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add velocity, frequency, and temporal features to enhance fraud signal quality."""
    df = df.copy()
    
    # Ensure timestamp
    if 'timestamp' not in df.columns:
        raise ValueError("Timestamp column required for feature engineering.")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time-based components
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    
    # Transaction velocity per user
    if 'user_id' in df.columns:
        df['txns_per_user_hour'] = df.groupby(['user_id', 'hour'])['transaction_id'].transform('count')
        df['txns_per_user_day'] = df.groupby(['user_id', 'day'])['transaction_id'].transform('count')
    else:
        df['txns_per_user_hour'] = 0
        df['txns_per_user_day'] = 0

    # Frequency features (card/address)
    for col in ['card1', 'addr1']:
        if col in df.columns:
            df[f'{col}_freq'] = df.groupby(col)['transaction_id'].transform('count')
        else:
            df[f'{col}_freq'] = 0

    # Rolling transaction count (1h window)
    try:
        df = df.sort_values(['user_id', 'timestamp'])
        df['rolling_txn_count_1h'] = (
            df.groupby('user_id')
            .rolling('1h', on='timestamp')['transaction_id']
            .count()
            .reset_index(level=0, drop=True)
        )
    except Exception as e:
        print(f"[WARN] rolling_txn_count_1h failed: {e}")
        df['rolling_txn_count_1h'] = 0

    print('[INFO] Added advanced temporal, velocity, and frequency features.')
    return df

# Data Drift Detection
def detect_data_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, float]:
    """Detect simple distributional drift using relative mean change per numeric feature.
    Returns dict of features whose relative mean change exceeds threshold.
    """
    drift_report: Dict[str, float] = {}
    num_cols = [c for c in reference_df.select_dtypes(include=[np.number]).columns if c in new_df.columns]
    for col in num_cols:
        ref_mean = reference_df[col].mean()
        new_mean = new_df[col].mean()
        denom = abs(ref_mean) + 1e-8
        rel_change = abs(ref_mean - new_mean) / denom
        drift_report[col] = rel_change

    significant = {k: v for k, v in drift_report.items() if v > threshold}
    if significant:
        print('[ALERT] Data drift detected for features:', significant)
    else:
        print('[INFO] No significant drift detected.')
    return significant

# Streaming Simulation
def simulate_stream(df: pd.DataFrame, batch_size: int = 100, delay: float = 1.0) -> Iterable[pd.DataFrame]:
    """Yield mini-batches to simulate streaming."""
    total = len(df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]
        yield batch
        print(f"[STREAM] Sent batch {start // batch_size + 1} ({len(batch)} transactions)")
        time.sleep(delay)

# Metadata Logging (MLOps)
def log_data_ingestion(metadata: Dict[str, Any], path: str = 'logs/data_ingestion_log.json') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata = metadata.copy()
    metadata['timestamp'] = datetime.utcnow().isoformat()
    with open(path, 'a') as f:
        f.write(json.dumps(metadata) + '\n')
    print(f"[INFO] Logged ingestion metadata to {path}")

# Simple fraud injection for RL / simulation
def simulate_fraud_events(df: pd.DataFrame, fraud_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Inject synthetic fraud labels into dataset for RL training or testing."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    df['is_fraud_sim'] = rng.choice([0, 1], size=len(df), p=[1 - fraud_rate, fraud_rate])
    print(f"[INFO] Injected synthetic fraud labels at rate={fraud_rate}")
    return df

# Save Master Dataset + artifacts
def save_master_dataset(df: pd.DataFrame, path: str = 'data\master_dataset.parquet') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[INFO] Master dataset saved at {path}.")

def save_artifacts(scaler: StandardScaler, df: pd.DataFrame, model_dir: str = 'models') -> None:
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    features_path = os.path.join(model_dir, 'feature_columns.joblib')
    joblib.dump(scaler, scaler_path)
    # feature columns (exclude label & transaction id)
    exclude_cols = ['Class', 'transaction_id', 'timestamp', 'Time', 'is_fraud_sim']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    joblib.dump(feature_cols, features_path)
    print(f"[INFO] Saved scaler -> {scaler_path} and feature columns -> {features_path}")

# -----------------------------
# CLI / Example Pipeline
# -----------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data\creditcard.csv', help='Path to raw CSV dataset')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--delay', type=float, default=0.1)
    parser.add_argument('--simulate-fraud', action='store_true')
    args = parser.parse_args()

    # 1. Load
    df_raw = load_dataset(args.path)

    # 2. Add synthetic environmental features
    df_en = add_synthetic_features(df_raw)

    # 3. Optionally inject synthetic fraud labels for simulation
    if args.simulate_fraud:
        df_enh = simulate_fraud_events(df_en, fraud_rate=0.01)
    else:
        df_enh = df_en  # ensure df_enh exists

    # 4. Preprocess and save master dataset
    # 4. Apply advanced feature engineering
    df_feat = feature_engineering(df_enh)
    
    # 5. Preprocess and save master dataset
    df_processed, fitted_scaler = preprocess_features(df_feat)

    save_master_dataset(df_processed, path='data/master_dataset.parquet')

    # Save enhanced dataset (raw + synthetic features)
    enhanced_path = "data\enhanced_creditcard.csv"
    os.makedirs(os.path.dirname(enhanced_path), exist_ok=True)
    df_en.to_csv(enhanced_path, index=False)
    print(f"[INFO] Enhanced dataset saved → {enhanced_path}")

    # Save scaler and feature columns for downstream scoring
    save_artifacts(fitted_scaler, df_processed, model_dir='models')

    # 5. Log ingestion metadata for MLOps/audit
    meta = {
        'source_path': args.path,
        'num_rows': len(df_processed),
        'batch_size': args.batch_size,
        'simulate_fraud': args.simulate_fraud,
        'features_added': [
            'hour', 'day', 'weekday',
            'txns_per_user_hour', 'txns_per_user_day',
            'card1_freq', 'addr1_freq',
            'rolling_txn_count_1h'
            ]
    }
    log_data_ingestion(meta)

    # 6. Stream one batch to demonstrate streaming
    stream_gen = simulate_stream(df_processed, batch_size=args.batch_size, delay=args.delay)
    for i, batch in enumerate(stream_gen):
        print(batch.head(2).to_string())
        if i >= 0:
            break
    print('[INFO] Pipeline finished.')


