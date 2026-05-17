"""
PainPrint - Synthetic Data Generator
Author: Rajat Mishra

Generates realistic smartphone behavioral data correlated with pain scores.
Each row = one hour of a user's phone behavior.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_user_data(user_id: int, n_days: int = 90) -> pd.DataFrame:
    """
    Generate synthetic behavioral data for one user over n_days.
    Pain scores follow a realistic chronic pain pattern with flares.
    """
    records = []
    base_pain = np.random.uniform(3, 6)  # Each user has a base pain level

    # Generate daily pain scores with flare patterns
    daily_pain = []
    pain = base_pain
    for d in range(n_days):
        # Random flare-ups
        if np.random.rand() < 0.1:
            pain = min(10, pain + np.random.uniform(2, 4))
        else:
            pain = max(0, pain + np.random.uniform(-0.5, 0.5))
        daily_pain.append(round(pain, 1))

    start_date = datetime(2024, 1, 1)

    for day_idx in range(n_days):
        pain_score = daily_pain[day_idx]
        pain_norm = pain_score / 10.0  # 0 to 1

        # Each day: generate hourly active-hour records (8am to 11pm = 16 hours)
        for hour in range(8, 24):
            timestamp = start_date + timedelta(days=day_idx, hours=hour)

            # Behavioral features — inversely correlated with pain
            noise = np.random.normal(0, 0.05)

            typing_speed_wpm = max(5, np.random.normal(
                45 - 20 * pain_norm, 5) + noise * 10)

            backspace_rate = max(0, np.random.normal(
                0.05 + 0.20 * pain_norm, 0.02))

            inter_key_delay_ms = max(50, np.random.normal(
                120 + 150 * pain_norm, 20))

            scroll_velocity = max(0, np.random.normal(
                800 - 400 * pain_norm, 80))

            app_switch_rate = max(0, np.random.normal(
                3 + 4 * pain_norm, 0.8))  # restlessness

            session_duration_min = max(1, np.random.normal(
                25 - 10 * pain_norm, 5))

            night_screen = 1 if (hour >= 22 and pain_score > 6) else 0
            if hour >= 22:
                night_screen = int(np.random.rand() < (0.1 + 0.4 * pain_norm))

            touch_pressure_norm = max(0, min(1, np.random.normal(
                0.5 - 0.2 * pain_norm, 0.1)))

            screen_brightness_pref = max(0, min(1, np.random.normal(
                0.7 - 0.2 * pain_norm, 0.1)))  # lower brightness in pain

            notification_response_delay_s = max(1, np.random.normal(
                30 + 60 * pain_norm, 10))

            records.append({
                'user_id': f'user_{user_id:03d}',
                'timestamp': timestamp,
                'day': day_idx,
                'hour': hour,
                'pain_score': pain_score,  # Ground truth label
                'typing_speed_wpm': round(typing_speed_wpm, 2),
                'backspace_rate': round(backspace_rate, 4),
                'inter_key_delay_ms': round(inter_key_delay_ms, 2),
                'scroll_velocity': round(scroll_velocity, 2),
                'app_switch_rate': round(app_switch_rate, 2),
                'session_duration_min': round(session_duration_min, 2),
                'night_screen_usage': night_screen,
                'touch_pressure_norm': round(touch_pressure_norm, 4),
                'screen_brightness_pref': round(screen_brightness_pref, 4),
                'notification_response_delay_s': round(notification_response_delay_s, 2),
            })

    return pd.DataFrame(records)


def generate_dataset(n_users: int = 50, n_days: int = 90) -> pd.DataFrame:
    print(f"Generating data for {n_users} users over {n_days} days...")
    all_data = []
    for uid in range(n_users):
        user_df = generate_user_data(uid, n_days)
        all_data.append(user_df)
        if (uid + 1) % 10 == 0:
            print(f"  ✓ {uid + 1}/{n_users} users done")
    df = pd.concat(all_data, ignore_index=True)
    print(f"Total records: {len(df):,}")
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = generate_dataset(n_users=50, n_days=90)
    df.to_csv("data/raw/behavioral_logs.csv", index=False)
    print("\n✅ Dataset saved to data/raw/behavioral_logs.csv")

    # Save small sample for quick demo
    sample = df[df['user_id'] == 'user_000'].head(200)
    sample.to_csv("data/sample_data.csv", index=False)
    print("✅ Sample saved to data/sample_data.csv")
    print(df.describe())
