import os
import argparse
import math
import pandas as pd
import numpy as np
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, values: np.ndarray, input_len: int, output_len: int):
        self.values = values.astype(np.float32)
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len

    def __len__(self) -> int:
        return max(0, len(self.values) - self.total_len + 1)

    def __getitem__(self, idx: int):
        window = self.values[idx : idx + self.total_len]
        x = window[: self.input_len]
        y = window[self.input_len : self.input_len + self.output_len]
        return torch.from_numpy(x), torch.from_numpy(y)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # [B, T, H]
        last = out[:, -1, :]   # [B, H]
        yhat = self.head(last) # [B, O]
        return yhat


def build_sequences(df: pd.DataFrame, target_col: str, group_col: str, input_len: int, output_len: int) -> Tuple[np.ndarray, List[str]]:
    df = df.copy()
    if 'datetime' in df.columns:
        df = df.sort_values(['station', 'datetime']) if 'station' in df.columns else df.sort_values('datetime')

    feature_cols = [c for c in df.columns if c not in {'datetime', group_col}]
    if target_col not in feature_cols:
        raise ValueError(f"target_col '{target_col}' not found among features")

    # Reorder to put target last; the model will predict only target
    non_target = [c for c in feature_cols if c != target_col]
    ordered = non_target + [target_col]

    arrays: List[np.ndarray] = []
    if group_col in df.columns:
        for _, g in df.groupby(group_col):
            arrays.append(g[ordered].to_numpy())
    else:
        arrays.append(df[ordered].to_numpy())

    series = np.concatenate(arrays, axis=0)
    return series, ordered


def split_train_val(series: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    n = len(series)
    n_val = int(math.floor(n * val_ratio))
    return series[:-n_val], series[-n_val:]


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, device: torch.device, lr: float = 1e-3) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        print(f"Epoch {epoch:03d} | train MSE: {train_loss/n_train:.6f} | val MSE: {val_loss/n_val:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train LSTM for air quality forecasting (24h -> 1h)")
    parser.add_argument('--data', type=str, default=os.path.join('air_quality_prediction', 'data', 'prepared', 'fused_imputed.csv'))
    parser.add_argument('--target', type=str, default='pm2.5')
    parser.add_argument('--group_col', type=str, default='station')
    parser.add_argument('--input_len', type=int, default=24)
    parser.add_argument('--output_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default=os.path.join('air_quality_prediction', 'models'))
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    # Normalize column names to lower-case for consistency
    df.columns = [c.lower() for c in df.columns]

    # Filter to numeric features and datetime/group columns
    keep_cols = [c for c in df.columns if c in {'datetime', args.group_col, args.target} or np.issubdtype(df[c].dtype, np.number)]
    df = df[keep_cols]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])

    series, ordered = build_sequences(df, target_col=args.target, group_col=args.group_col, input_len=args.input_len, output_len=args.output_len)

    # Scale features per full series (simple standardization)
    mean = series.mean(axis=0, keepdims=True)
    std = series.std(axis=0, keepdims=True) + 1e-8
    series_norm = (series - mean) / std

    train_series, val_series = split_train_val(series_norm, val_ratio=0.2)

    input_dim = series.shape[1]
    output_dim = args.output_len  # predict target 1-step ahead as scalar per horizon

    # Only feed features, but predict target; we kept target as last column
    # For supervision, y should be the target dimension only
    # We create datasets from the series but then slice y to last column
    train_ds = SequenceDataset(train_series, args.input_len, args.output_len)
    val_ds = SequenceDataset(val_series, args.input_len, args.output_len)

    def collate_only_target(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)  # [B, T, F]
        y = torch.stack(ys, dim=0)  # [B, O, F]
        # Take the target column (last feature) from each step and last horizon only
        # Reduce [B, O, F] -> [B, O] using last feature index
        y = y[..., -1]
        return x, y

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_only_target)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_only_target)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMRegressor(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout, output_dim=output_dim).to(device)

    train(model, train_loader, val_loader, epochs=args.epochs, device=device, lr=args.lr)

    save_path = os.path.join(args.save_dir, 'lstm_model.pt')
    torch.save({'state_dict': model.state_dict(), 'config': vars(args), 'feature_order': ordered, 'norm_mean': mean, 'norm_std': std}, save_path)
    print(f"Saved model to {save_path}")


if __name__ == '__main__':
    main()
