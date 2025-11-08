import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import torch
from torch import nn
import joblib
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load prepared data
DATA_PATH = os.path.join('data', 'prepared', 'fused_imputed.csv')
LSTM_MODEL_PATH = os.path.join('models', 'lstm_model.pt')
XGB_MODEL_PATH = os.path.join('models', 'xgb_model.pkl')


# Define LSTM model architecture (same as in train_lstm.py)
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


df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = [c.lower() for c in df.columns]

# Rename column for dashboard consistency
if 'station' in df.columns:
    df.rename(columns={'station': 'station_id'}, inplace=True)

# Ensure datetime
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
else:
    st.error("Column 'datetime' not found!")
    st.stop()


@st.cache_resource
def load_lstm_model():
    """Load LSTM model"""
    if not os.path.exists(LSTM_MODEL_PATH):
        return None, None
    
    try:
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        
        # Reconstruct model using the class defined above
        model = LSTMRegressor(
            input_dim=len(checkpoint['feature_order']),
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            output_dim=config['output_len']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        return model, checkpoint
    except Exception as e:
        st.error(f"Error loading LSTM model: {e}")
        return None, None


@st.cache_resource
def load_xgb_model():
    """Load XGBoost model"""
    if not os.path.exists(XGB_MODEL_PATH):
        return None
    
    try:
        checkpoint = joblib.load(XGB_MODEL_PATH)
        return checkpoint
    except Exception as e:
        st.error(f"Error loading XGBoost model: {e}")
        return None


def prepare_lstm_input(data, feature_order, input_len, norm_mean, norm_std):
    """Prepare input sequences for LSTM prediction"""
    # Get the features in correct order
    values = data[feature_order].values
    
    # Normalize
    values_norm = (values - norm_mean) / norm_std
    
    # Create sequences
    sequences = []
    for i in range(len(values_norm) - input_len):
        sequences.append(values_norm[i:i+input_len])
    
    if len(sequences) == 0:
        return None
    
    return torch.FloatTensor(np.array(sequences))


def prepare_xgb_input(data, feature_cols, input_len):
    """Prepare lag features for XGBoost prediction"""
    values = data[feature_cols].values
    
    X = []
    for i in range(len(values) - input_len):
        X.append(values[i:i+input_len].flatten())
    
    if len(X) == 0:
        return None
    
    return np.array(X, dtype=np.float32)


def make_predictions(filtered_df, lstm_model, lstm_checkpoint, xgb_checkpoint):
    """Generate predictions from both models"""
    predictions = {'lstm': None, 'xgb': None, 'timestamps': None}
    
    if lstm_model is not None and lstm_checkpoint is not None:
        try:
            feature_order = lstm_checkpoint['feature_order']
            config = lstm_checkpoint['config']
            input_len = config['input_len']
            norm_mean = lstm_checkpoint['norm_mean']
            norm_std = lstm_checkpoint['norm_std']
            
            # Prepare input
            X_lstm = prepare_lstm_input(filtered_df, feature_order, input_len, norm_mean, norm_std)
            
            if X_lstm is not None:
                with torch.no_grad():
                    preds_norm = lstm_model(X_lstm).numpy().flatten()
                
                # Denormalize predictions (target is last in feature_order)
                target_idx = len(feature_order) - 1
                preds = preds_norm * norm_std[0, target_idx] + norm_mean[0, target_idx]
                
                # Timestamps for predictions (shifted by input_len)
                timestamps = filtered_df['datetime'].iloc[input_len:input_len+len(preds)].values
                
                predictions['lstm'] = preds
                predictions['timestamps'] = timestamps
        except Exception as e:
            st.warning(f"LSTM prediction error: {e}")
    
    if xgb_checkpoint is not None:
        try:
            model = xgb_checkpoint['model']
            scaler = xgb_checkpoint['scaler']
            feature_cols = xgb_checkpoint['feature_cols']
            
            # Use same input_len as LSTM if available, else default to 24
            input_len = lstm_checkpoint['config']['input_len'] if lstm_checkpoint else 24
            
            # Prepare input
            X_xgb = prepare_xgb_input(filtered_df, feature_cols, input_len)
            
            if X_xgb is not None:
                X_xgb_scaled = scaler.transform(X_xgb)
                preds = model.predict(X_xgb_scaled)
                
                if predictions['timestamps'] is None:
                    timestamps = filtered_df['datetime'].iloc[input_len:input_len+len(preds)].values
                    predictions['timestamps'] = timestamps
                
                predictions['xgb'] = preds
        except Exception as e:
            st.warning(f"XGBoost prediction error: {e}")
    
    return predictions


st.title("Beijing Air Quality Dashboard 🌆")
st.write("Interactive visualization of PM2.5 and weather data with ML predictions")

# Load models
st.sidebar.header("Model Status")
lstm_model, lstm_checkpoint = load_lstm_model()
xgb_checkpoint = load_xgb_model()

if lstm_model is not None:
    st.sidebar.success("✅ LSTM Model Loaded")
else:
    st.sidebar.warning("⚠️ LSTM Model Not Available")

if xgb_checkpoint is not None:
    st.sidebar.success("✅ XGBoost Model Loaded")
else:
    st.sidebar.warning("⚠️ XGBoost Model Not Available")

# Sidebar filters
st.sidebar.header("Filters")
stations = df['station_id'].unique()
selected_station = st.sidebar.selectbox("Select station:", stations)

start_date = st.sidebar.date_input("Start date", df['datetime'].min().date())
end_date = st.sidebar.date_input("End date", df['datetime'].max().date())

mask = (
    (df['station_id'] == selected_station) &
    (df['datetime'].dt.date >= start_date) &
    (df['datetime'].dt.date <= end_date)
)
filtered_df = df.loc[mask].reset_index(drop=True)

# Generate predictions
predictions = make_predictions(filtered_df, lstm_model, lstm_checkpoint, xgb_checkpoint)

# Time series plot with predictions
st.subheader("PM2.5 over time with Model Predictions")

fig_pm25 = go.Figure()

# Actual values
fig_pm25.add_trace(go.Scatter(
    x=filtered_df['datetime'],
    y=filtered_df['pm2.5'],
    mode='lines',
    name='Actual PM2.5',
    line=dict(color='blue', width=2)
))

# LSTM predictions
if predictions['lstm'] is not None:
    fig_pm25.add_trace(go.Scatter(
        x=predictions['timestamps'],
        y=predictions['lstm'],
        mode='lines',
        name='LSTM Prediction',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Calculate LSTM metrics
    actual_values = filtered_df['pm2.5'].iloc[lstm_checkpoint['config']['input_len']:lstm_checkpoint['config']['input_len']+len(predictions['lstm'])].values
    lstm_mae = np.mean(np.abs(actual_values - predictions['lstm']))
    lstm_rmse = np.sqrt(np.mean((actual_values - predictions['lstm'])**2))
    st.sidebar.metric("LSTM MAE", f"{lstm_mae:.2f}")
    st.sidebar.metric("LSTM RMSE", f"{lstm_rmse:.2f}")

# XGBoost predictions
if predictions['xgb'] is not None:
    fig_pm25.add_trace(go.Scatter(
        x=predictions['timestamps'],
        y=predictions['xgb'],
        mode='lines',
        name='XGBoost Prediction',
        line=dict(color='green', width=2, dash='dot')
    ))
    
    # Calculate XGBoost metrics
    input_len = lstm_checkpoint['config']['input_len'] if lstm_checkpoint else 24
    actual_values = filtered_df['pm2.5'].iloc[input_len:input_len+len(predictions['xgb'])].values
    xgb_mae = np.mean(np.abs(actual_values - predictions['xgb']))
    xgb_rmse = np.sqrt(np.mean((actual_values - predictions['xgb'])**2))
    st.sidebar.metric("XGBoost MAE", f"{xgb_mae:.2f}")
    st.sidebar.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")

fig_pm25.update_layout(
    title=f"PM2.5 at {selected_station}",
    xaxis_title="Date",
    yaxis_title="PM2.5 (µg/m³)",
    hovermode='x unified',
    height=500
)
st.plotly_chart(fig_pm25, use_container_width=True)

# Prediction comparison metrics
if predictions['lstm'] is not None and predictions['xgb'] is not None:
    st.subheader("Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    input_len = lstm_checkpoint['config']['input_len'] if lstm_checkpoint else 24
    actual_values = filtered_df['pm2.5'].iloc[input_len:input_len+len(predictions['lstm'])].values
    
    with col1:
        st.metric("Actual Mean PM2.5", f"{np.mean(actual_values):.2f} µg/m³")
    
    with col2:
        lstm_mean = np.mean(predictions['lstm'])
        lstm_diff = lstm_mean - np.mean(actual_values)
        st.metric("LSTM Mean Prediction", f"{lstm_mean:.2f} µg/m³", f"{lstm_diff:+.2f}")
    
    with col3:
        xgb_mean = np.mean(predictions['xgb'])
        xgb_diff = xgb_mean - np.mean(actual_values)
        st.metric("XGBoost Mean Prediction", f"{xgb_mean:.2f} µg/m³", f"{xgb_diff:+.2f}")
    
    # Scatter plot: Actual vs Predicted
    fig_scatter = make_subplots(
        rows=1, cols=2,
        subplot_titles=("LSTM: Actual vs Predicted", "XGBoost: Actual vs Predicted")
    )
    
    # LSTM scatter
    fig_scatter.add_trace(
        go.Scatter(x=actual_values, y=predictions['lstm'], mode='markers',
                   name='LSTM', marker=dict(color='red', size=5, opacity=0.6)),
        row=1, col=1
    )
    fig_scatter.add_trace(
        go.Scatter(x=[actual_values.min(), actual_values.max()],
                   y=[actual_values.min(), actual_values.max()],
                   mode='lines', name='Perfect Prediction',
                   line=dict(color='black', dash='dash')),
        row=1, col=1
    )
    
    # XGBoost scatter
    fig_scatter.add_trace(
        go.Scatter(x=actual_values, y=predictions['xgb'], mode='markers',
                   name='XGBoost', marker=dict(color='green', size=5, opacity=0.6)),
        row=1, col=2
    )
    fig_scatter.add_trace(
        go.Scatter(x=[actual_values.min(), actual_values.max()],
                   y=[actual_values.min(), actual_values.max()],
                   mode='lines', name='Perfect Prediction',
                   line=dict(color='black', dash='dash'), showlegend=False),
        row=1, col=2
    )
    
    fig_scatter.update_xaxes(title_text="Actual PM2.5 (µg/m³)", row=1, col=1)
    fig_scatter.update_xaxes(title_text="Actual PM2.5 (µg/m³)", row=1, col=2)
    fig_scatter.update_yaxes(title_text="Predicted PM2.5 (µg/m³)", row=1, col=1)
    fig_scatter.update_yaxes(title_text="Predicted PM2.5 (µg/m³)", row=1, col=2)
    
    fig_scatter.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Other pollutants / weather features
st.subheader("Other pollutants / weather features")
numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in {'pm2.5'}]

if numeric_cols:
    feature_to_plot = st.selectbox("Select feature to plot:", numeric_cols)
    fig_feat = px.line(filtered_df, x='datetime', y=feature_to_plot, title=f"{feature_to_plot} at {selected_station}")
    st.plotly_chart(fig_feat, use_container_width=True)
else:
    st.write("No numeric features available to plot.")

# Optional: Correlation heatmap
st.subheader("Correlation heatmap")
corr = filtered_df[numeric_cols + ['pm2.5']].corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
st.plotly_chart(fig_corr, use_container_width=True)
