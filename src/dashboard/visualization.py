import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Load prepared data
DATA_PATH = os.path.join('air_quality_prediction', 'data', 'prepared', 'fused_imputed.csv')
df = pd.read_csv(DATA_PATH)

# Rename column for dashboard consistency
if 'station' in df.columns:
    df.rename(columns={'station': 'station_id'}, inplace=True)

# Ensure datetime
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
else:
    st.error("Column 'datetime' not found!")
    st.stop()

st.title("Beijing Air Quality Dashboard 🌆")
st.write("Interactive visualization of PM2.5 and weather data")

# Sidebar filters
stations = df['station_id'].unique()
selected_station = st.sidebar.selectbox("Select station:", stations)

start_date = st.sidebar.date_input("Start date", df['datetime'].min().date())
end_date = st.sidebar.date_input("End date", df['datetime'].max().date())

mask = (
    (df['station_id'] == selected_station) &
    (df['datetime'].dt.date >= start_date) &
    (df['datetime'].dt.date <= end_date)
)
filtered_df = df.loc[mask]

# Time series plot
st.subheader("PM2.5 over time")
fig_pm25 = px.line(filtered_df, x='datetime', y='pm2.5', title=f"PM2.5 at {selected_station}")
st.plotly_chart(fig_pm25, use_container_width=True)

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
