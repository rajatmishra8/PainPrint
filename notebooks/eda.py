"""
PainPrint - Exploratory Data Analysis
Author: Rajat Mishra

Run this script to generate all EDA plots and statistics.
Equivalent to notebooks/01_EDA.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'behavioral_logs.csv')

if not os.path.exists(DATA_PATH):
    print("Generating data first...")
    import subprocess
    subprocess.run(['python', os.path.join(os.path.dirname(__file__), '..', 'src', 'data_generator.py')])

df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
print(f"Dataset shape: {df.shape}")
print(df.describe().round(2))

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

BEHAVIORAL_COLS = [
    'typing_speed_wpm', 'backspace_rate', 'inter_key_delay_ms',
    'scroll_velocity', 'app_switch_rate', 'session_duration_min',
    'touch_pressure_norm', 'screen_brightness_pref',
    'notification_response_delay_s'
]

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor']   = '#16213e'
plt.rcParams['text.color']       = 'white'
plt.rcParams['axes.labelcolor']  = 'white'
plt.rcParams['xtick.color']      = 'white'
plt.rcParams['ytick.color']      = 'white'

# ─────────────────────────────────────────────────────────────────────────────
# 1. Pain Score Distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(df['pain_score'], bins=40, color='#7EB8F7', edgecolor='none', alpha=0.8)
ax.axvline(df['pain_score'].mean(), color='#F47E7E', linestyle='--', label=f'Mean: {df["pain_score"].mean():.2f}')
ax.set_title('Distribution of Pain Scores Across All Users & Days', fontsize=13)
ax.set_xlabel('Pain Score (0–10)')
ax.set_ylabel('Frequency')
ax.legend()
fig.suptitle('PainPrint EDA | Rajat Mishra', fontsize=10, color='#aaa')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_pain_distribution.png', dpi=150)
print("✅ Saved: eda_pain_distribution.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Correlations with Pain
# ─────────────────────────────────────────────────────────────────────────────
corrs = df[BEHAVIORAL_COLS + ['pain_score']].corr()['pain_score'].drop('pain_score').sort_values()
colors = ['#F47E7E' if c > 0 else '#7EB8F7' for c in corrs]

fig, ax = plt.subplots(figsize=(10, 5))
corrs.plot(kind='barh', color=colors, ax=ax)
ax.axvline(0, color='white', linewidth=0.8)
ax.set_title('Behavioral Feature Correlation with Pain Score', fontsize=13)
ax.set_xlabel('Pearson Correlation')
fig.suptitle('PainPrint EDA | Rajat Mishra', fontsize=10, color='#aaa')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_correlations.png', dpi=150)
print("✅ Saved: eda_correlations.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Circadian Pain Pattern (avg by hour)
# ─────────────────────────────────────────────────────────────────────────────
hourly = df.groupby('hour')['pain_score'].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(hourly.index, hourly.values, color='#A77EF4', linewidth=2.5, marker='o', markersize=5)
ax.fill_between(hourly.index, hourly.values, alpha=0.2, color='#A77EF4')
ax.set_title('Average Pain Score by Hour of Day (Circadian Pattern)', fontsize=13)
ax.set_xlabel('Hour')
ax.set_ylabel('Avg Pain Score')
ax.set_xticks(range(8, 24))
fig.suptitle('PainPrint EDA | Rajat Mishra', fontsize=10, color='#aaa')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_circadian.png', dpi=150)
print("✅ Saved: eda_circadian.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Single-User Pain Timeline (user_000)
# ─────────────────────────────────────────────────────────────────────────────
u0 = df[df['user_id'] == 'user_000'].groupby('day')['pain_score'].mean()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(u0.index, u0.values, color='#7EB8F7', linewidth=1.8)
ax.fill_between(u0.index, u0.values, alpha=0.15, color='#7EB8F7')
ax.axhline(7, color='#F47E7E', linestyle='--', alpha=0.7, label='Flare Threshold (7)')
ax.set_title('User_000: Daily Pain Score Over 90 Days', fontsize=13)
ax.set_xlabel('Day')
ax.set_ylabel('Avg Daily Pain Score')
ax.legend()
fig.suptitle('PainPrint EDA | Rajat Mishra', fontsize=10, color='#aaa')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/eda_user_timeline.png', dpi=150)
print("✅ Saved: eda_user_timeline.png")
plt.close()

print("\n✅ All EDA plots saved to models/ directory.")
