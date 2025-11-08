import os
import argparse
import math
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib


def build_features_targets(df: pd.DataFrame, target_col: str, group_col: str, input_len: int, output_len: int = 1):
    """
    Build lag features for XGBoost: use previous `input_len` values to predict `output_len` ahead.
    Returns X, y arrays.
    """
    df = df.copy()
    if 'datetime' in df.columns:
        df = df.sort_values(['station', 'datetime']) if 'station' in df.columns else df.sort_values('datetime')

    feature_cols = [c for c in df.columns if c not in {'datetime', group_col}]
    
    if target_col not in feature_cols:
        raise ValueError(f"Target column '{target_col}' not found among features")

    # Only numeric columns
    feature_cols = [c for c in feature_cols if np.issubdtype(df[c].dtype, np.number)]

    X_list = []
    y_list = []

    if group_col in df.columns:
        groups = df[group_col].unique()
        for g in groups:
            sub = df[df[group_col] == g].copy()
            values = sub[feature_cols].values
            for i in range(len(values) - input_len - output_len + 1):
                X_list.append(values[i:i+input_len].flatten())
                y_list.append(values[i+input_len:i+input_len+output_len, feature_cols.index(target_col)])
    else:
        values = df[feature_cols].values
        for i in range(len(values) - input_len - output_len + 1):
            X_list.append(values[i:i+input_len].flatten())
            y_list.append(values[i+input_len:i+input_len+output_len, feature_cols.index(target_col)])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost for air quality forecasting")
    parser.add_argument('--data', type=str, default=os.path.join('data', 'prepared', 'fused_imputed.csv'))
    parser.add_argument('--target', type=str, default='pm2.5')
    parser.add_argument('--group_col', type=str, default='station')
    parser.add_argument('--input_len', type=int, default=24)
    parser.add_argument('--output_len', type=int, default=1)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--save_dir', type=str, default=os.path.join('models'))
    parser.add_argument('--n_estimators', type=int, default=500)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = [c.lower() for c in df.columns]

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])

    X, y, feature_cols = build_features_targets(df, target_col=args.target, group_col=args.group_col, input_len=args.input_len, output_len=args.output_len)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, shuffle=False)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective='reg:squarederror',
        tree_method='hist',  # faster for large datasets
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb.callback.EarlyStopping(rounds=50)], verbose=True)

    # Evaluate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Validation MSE: {mse:.6f}")

    # Save model and scaler
    save_path = os.path.join(args.save_dir, 'xgb_model.pkl')
    joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols}, save_path)
    print(f"Saved XGBoost model and scaler to {save_path}")


if __name__ == '__main__':
    main()
