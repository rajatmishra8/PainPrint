"""
PainPrint - Feature Engineering
Author: Rajat Mishra

Transforms raw hourly behavioral logs into structured sequences for LSTM input.
Computes rolling statistics, circadian features, and the Digital Pain Signature.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


BEHAVIORAL_COLS = [
    'typing_speed_wpm',
    'backspace_rate',
    'inter_key_delay_ms',
    'scroll_velocity',
    'app_switch_rate',
    'session_duration_min',
    'night_screen_usage',
    'touch_pressure_norm',
    'screen_brightness_pref',
    'notification_response_delay_s',
]


def add_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Add rolling mean and std of behavioral features per user."""
    df = df.sort_values(['user_id', 'timestamp']).copy()
    for col in BEHAVIORAL_COLS:
        df[f'{col}_roll_mean'] = (
            df.groupby('user_id')[col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        df[f'{col}_roll_std'] = (
            df.groupby('user_id')[col]
            .transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        )
    return df


def add_circadian_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour of day as sine/cosine for cyclical representation."""
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def add_pain_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Digital Pain Signature: deviation from user's own baseline.
    This captures intra-user behavioral changes, not inter-user differences.
    """
    for col in BEHAVIORAL_COLS:
        user_mean = df.groupby('user_id')[col].transform('mean')
        user_std = df.groupby('user_id')[col].transform('std').replace(0, 1)
        df[f'{col}_zscore'] = (df[col] - user_mean) / user_std
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")
    df = add_rolling_features(df)
    df = add_circadian_features(df)
    df = add_pain_deviation(df)
    df = df.dropna()
    print(f"  Features engineered. Shape: {df.shape}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    base = BEHAVIORAL_COLS.copy()
    roll = [c for c in df.columns if '_roll_mean' in c or '_roll_std' in c]
    circ = ['hour_sin', 'hour_cos']
    zscore = [c for c in df.columns if '_zscore' in c]
    return base + roll + circ + zscore


def build_sequences(df: pd.DataFrame, seq_len: int = 16, target_col: str = 'pain_score'):
    """
    Build (X, y) sequences per user.
    Each sequence = seq_len hours of behavior → next hour's pain score.
    """
    feature_cols = get_feature_cols(df)
    X_all, y_all, user_ids = [], [], []

    for uid, group in df.groupby('user_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        features = group[feature_cols].values
        targets = group[target_col].values

        for i in range(len(group) - seq_len):
            X_all.append(features[i:i+seq_len])
            y_all.append(targets[i+seq_len])
            user_ids.append(uid)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)
    print(f"  Sequences built: X={X.shape}, y={y.shape}")
    return X, y, user_ids, feature_cols


def scale_and_save(X_train, X_test, save_dir='models'):
    """Fit scaler on train, transform both."""
    os.makedirs(save_dir, exist_ok=True)
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)
    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, seq_len, n_feat)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape[0], seq_len, n_feat)
    joblib.dump(scaler, f'{save_dir}/scaler.pkl')
    print(f"  Scaler saved to {save_dir}/scaler.pkl")
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    df = pd.read_csv("data/raw/behavioral_logs.csv", parse_dates=['timestamp'])
    df = engineer_features(df)
    df.to_csv("data/processed/engineered_features.csv", index=False)
    print("✅ Saved to data/processed/engineered_features.csv")
