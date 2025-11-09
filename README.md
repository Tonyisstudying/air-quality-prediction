# Air Quality Prediction System

A machine learning project for predicting PM2.5 air pollution levels in Beijing using LSTM and XGBoost models. This system analyzes historical air quality and meteorological data from 12 monitoring stations across Beijing (2013-2017) to forecast future PM2.5 concentrations.

## Introduction

Air pollution is a critical public health concern, particularly in urban areas. This project implements two complementary machine learning approaches:

- **LSTM (Long Short-Term Memory)**: A deep learning model that captures temporal dependencies and sequential patterns in time series data
- **XGBoost**: A gradient boosting model using engineered lag features for robust predictions

The system processes hourly measurements of pollutants (PM2.5, PM10, SO2, NO2, CO, O3) and weather variables (temperature, pressure, humidity, wind) to predict PM2.5 levels. An interactive dashboard visualizes both historical data and real-time model predictions.

## Features

✅ Data preprocessing with KNN imputation for missing values  
✅ LSTM neural network for sequence-based forecasting  
✅ XGBoost regression with lag feature engineering  
✅ Interactive Streamlit dashboard with prediction visualization  
✅ Performance metrics (MAE, RMSE) for model comparison  
✅ Multi-station support across Beijing  

## Project Structure

```
air-quality-prediction/
├── data/
│   ├── raw/                    # Original CSV files from 12 stations
│   └── prepared/               # Preprocessed and imputed data
│       └── fused_imputed.csv
├── models/
│   ├── lstm_model.pt          # Trained LSTM model
│   └── xgb_model.pkl          # Trained XGBoost model
├── src/
│   ├── prepare/
│   │   └── prepare_data.py    # Data loading and preprocessing
│   ├── models/
│   │   ├── train_lstm.py      # LSTM training script
│   │   └── train_xgb.py       # XGBoost training script
│   ├── dashboard/
│   │   └── visualization.py   # Interactive dashboard
│   └── utils/
│       └── visualize_architecture.py  # Model architecture diagrams
└── reports/
    └── section_3_lstm.md      # Model documentation
```

## Installation

### Prerequisites

- Python 3.8+
- Git

### Clone the Repository

```bash
git clone https://github.com/Tonyisstudying/air-quality-prediction.git
cd air-quality-prediction
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn torch xgboost joblib streamlit plotly matplotlib
```

Or create a `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Data Preparation

Preprocess raw data and handle missing values:

```bash
python src\prepare\prepare_data.py
```

**Output:** `data/prepared/fused_imputed.csv`

### 2. Train Models

#### Train LSTM Model

```bash
python src\models\train_lstm.py
```

Optional arguments:
```bash
python src\models\train_lstm.py --epochs 20 --batch_size 128 --hidden_dim 128 --lr 0.001
```

**Output:** `models/lstm_model.pt`

#### Train XGBoost Model

```bash
python src\models\train_xgb.py
```

Optional arguments:
```bash
python src\models\train_xgb.py --n_estimators 500 --max_depth 6 --learning_rate 0.05
```

**Output:** `models/xgb_model.pkl`

### 3. Run Dashboard

Launch the interactive visualization dashboard:

```bash
python -m streamlit run src\dashboard\visualization.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Dashboard Features:**
- Select monitoring station
- Choose date range
- View actual vs predicted PM2.5 levels
- Compare LSTM and XGBoost performance
- Explore pollutant correlations
- Analyze other weather features

## Model Performance

| Model    | MAE (µg/m³) | RMSE (µg/m³) | Description |
|----------|-------------|--------------|-------------|
| LSTM     | ~45-55      | ~65-75       | Captures temporal patterns over 24-hour windows |
| XGBoost  | ~40-50      | ~60-70       | Robust to outliers with lag features |

*Note: Performance varies by station and time period*

## Architecture

### LSTM Model
- **Input:** 24-hour sequence with 16 features
- **Architecture:** 2 LSTM layers (128 hidden units each) + MLP head
- **Output:** Single PM2.5 prediction for next hour
- **Training:** Adam optimizer, MSE loss, 20 epochs

### XGBoost Model
- **Input:** Flattened 24-hour lag features
- **Parameters:** 500 trees, max depth 6, learning rate 0.05
- **Output:** PM2.5 regression
- **Training:** Early stopping on validation set

## Dataset

**Source:** Beijing Multi-Site Air Quality Dataset  
**Period:** March 2013 - February 2017  
**Frequency:** Hourly measurements  
**Stations:** 12 monitoring sites across Beijing  
**Variables:** PM2.5, PM10, SO2, NO2, CO, O3, temperature, pressure, dewpoint, rain, wind direction, wind speed

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request


## Acknowledgments

- Beijing Municipal Environmental Monitoring Center for the dataset
- PyTorch and XGBoost communities for excellent ML frameworks
- Streamlit for the interactive dashboard framework
---
