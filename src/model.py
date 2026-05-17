"""
PainPrint - LSTM Pain Prediction Model
Author: Rajat Mishra

Trains an LSTM model to predict pain scores from behavioral sequences.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features, build_sequences, scale_and_save


# ── Model ────────────────────────────────────────────────────────────────────

class PainPrintLSTM(nn.Module):
    def __init__(self, input_size: int, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.fc1   = nn.Linear(hidden2, 32)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])   # last time step
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val,
                n_epochs=30, batch_size=64, lr=1e-3, save_dir='models'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)

    model = PainPrintLSTM(input_size=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train MSE: {avg_train:.4f} | Val MSE: {val_loss:.4f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{save_dir}/lstm_painprint.pt')
    print(f"\n✅ Model saved to {save_dir}/lstm_painprint.pt")
    return model, train_losses, val_losses, device


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, device):
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    y = y_test

    mae  = np.mean(np.abs(preds - y))
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n📊 Test Results:")
    print(f"   MAE  : {mae:.3f}")
    print(f"   RMSE : {rmse:.3f}")
    print(f"   R²   : {r2:.3f}")
    return preds, mae, rmse, r2


def plot_results(train_losses, val_losses, y_test, preds, save_dir='models'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PainPrint — LSTM Results | Rajat Mishra', fontsize=14, fontweight='bold')

    # Loss curve
    axes[0].plot(train_losses, label='Train MSE', color='#2196F3')
    axes[0].plot(val_losses, label='Val MSE', color='#F44336')
    axes[0].set_title('Training Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Prediction vs actual
    idx = np.arange(min(200, len(y_test)))
    axes[1].plot(idx, y_test[:200], label='Actual Pain Score', color='#F44336', alpha=0.7)
    axes[1].plot(idx, preds[:200], label='Predicted', color='#2196F3', linestyle='--', alpha=0.9)
    axes[1].set_title('Prediction vs Actual (first 200 samples)')
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Pain Score (0-10)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/results.png', dpi=150)
    print(f"✅ Plot saved to {save_dir}/results.png")
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  PainPrint LSTM Training Pipeline | Rajat Mishra")
    print("=" * 55)

    # Load data
    raw_path = "data/raw/behavioral_logs.csv"
    if not os.path.exists(raw_path):
        print("Generating data first...")
        import subprocess
        subprocess.run(["python", "src/data_generator.py"])

    df = pd.read_csv(raw_path, parse_dates=['timestamp'])
    print(f"\nLoaded {len(df):,} records")

    # Feature engineering
    df = engineer_features(df)
    X, y, user_ids, feat_cols = build_sequences(df, seq_len=16)
    print(f"Feature columns used: {len(feat_cols)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42)

    # Scale
    X_train_s, X_test_s, scaler = scale_and_save(X_train, X_test)
    _, X_val_s, _ = scale_and_save(X_train, X_val)

    # Train
    print("\nTraining LSTM model...")
    model, train_losses, val_losses, device = train_model(
        X_train_s, y_train, X_val_s, y_val, n_epochs=30)

    # Evaluate
    preds, mae, rmse, r2 = evaluate(model, X_test_s, y_test, device)

    # Plot
    plot_results(train_losses, val_losses, y_test, preds)
