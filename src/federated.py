"""
PainPrint - Federated Learning Simulation
Author: Rajat Mishra

Simulates federated training across multiple users (clients).
Each client trains locally; only model weights are shared (FedAvg).
Raw behavioral data never leaves the simulated device.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import copy, os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features, build_sequences, scale_and_save
from model import PainPrintLSTM


def get_client_data(df, user_id, seq_len=16, feat_cols=None):
    """Extract sequence data for a single federated client (user)."""
    user_df = df[df['user_id'] == user_id].sort_values('timestamp')
    if len(user_df) < seq_len + 5:
        return None, None

    features = user_df[feat_cols].values.astype(np.float32)
    targets  = user_df['pain_score'].values.astype(np.float32)

    X = np.array([features[i:i+seq_len] for i in range(len(features)-seq_len)])
    y = np.array([targets[i+seq_len] for i in range(len(features)-seq_len)])
    return X, y


def local_train(model, X, y, epochs=3, lr=1e-3, device='cpu'):
    """Train model locally on one client's data."""
    model = model.to(device)
    model.train()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    return model.state_dict()


def fedavg(global_weights, client_weights_list, client_sizes):
    """Federated Averaging: weighted average of client model weights."""
    total = sum(client_sizes)
    avg_weights = copy.deepcopy(global_weights)
    for key in avg_weights:
        avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
        for cw, sz in zip(client_weights_list, client_sizes):
            avg_weights[key] += cw[key].float() * (sz / total)
    return avg_weights


def evaluate_global(model, X_test, y_test, device):
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(Xt).cpu().numpy()
    mae  = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    return mae, rmse


def run_federated(df, feat_cols, n_rounds=10, n_clients_per_round=10, device='cpu'):
    print("=" * 55)
    print("  PainPrint Federated Learning | Rajat Mishra")
    print("=" * 55)

    all_users = df['user_id'].unique().tolist()
    input_size = len(feat_cols)

    global_model = PainPrintLSTM(input_size=input_size).to(device)
    global_weights = global_model.state_dict()

    # Hold-out global test set
    X_all, y_all, _, _ = build_sequences(df, seq_len=16)
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.15, random_state=99)
    X_test_s = X_test  # simplified: no scaling for demo

    round_maes = []

    for rnd in range(1, n_rounds + 1):
        selected = np.random.choice(all_users,
                                    size=min(n_clients_per_round, len(all_users)),
                                    replace=False)
        client_weights = []
        client_sizes   = []

        for uid in selected:
            X_c, y_c = get_client_data(df, uid, feat_cols=feat_cols)
            if X_c is None or len(X_c) < 10:
                continue
            local_model = PainPrintLSTM(input_size=input_size)
            local_model.load_state_dict(copy.deepcopy(global_weights))
            w = local_train(local_model, X_c, y_c, epochs=3, device=device)
            client_weights.append(w)
            client_sizes.append(len(X_c))

        if not client_weights:
            continue

        global_weights = fedavg(global_weights, client_weights, client_sizes)
        global_model.load_state_dict(global_weights)

        mae, rmse = evaluate_global(global_model, X_test_s, y_test, device)
        round_maes.append(mae)
        print(f"  Round {rnd:2d}/{n_rounds} | Clients: {len(client_weights):2d} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    print(f"\n✅ Federated Training Complete!")
    print(f"   Final MAE  : {round_maes[-1]:.3f}")
    print(f"   Improvement: {round_maes[0] - round_maes[-1]:.3f} over {n_rounds} rounds")
    torch.save(global_model.state_dict(), 'models/lstm_federated.pt')
    print("   Model saved: models/lstm_federated.pt")
    return global_model, round_maes


if __name__ == "__main__":
    raw_path = "data/raw/behavioral_logs.csv"
    if not os.path.exists(raw_path):
        import subprocess; subprocess.run(["python", "src/data_generator.py"])

    df = pd.read_csv(raw_path, parse_dates=['timestamp'])
    df = engineer_features(df)

    from feature_engineering import get_feature_cols
    feat_cols = get_feature_cols(df)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('models', exist_ok=True)

    model, maes = run_federated(df, feat_cols, n_rounds=10, n_clients_per_round=10, device=device)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(maes)+1), maes, marker='o', color='#2196F3')
    plt.title('Federated Learning: MAE per Round | Rajat Mishra')
    plt.xlabel('Round'); plt.ylabel('MAE (Pain Score)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/federated_convergence.png', dpi=150)
    print("✅ Plot saved: models/federated_convergence.png")
