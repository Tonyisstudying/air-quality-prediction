# Prepare data
python src\prepare\prepare_data.py

# Train models
python src\models\train_lstm.py
python src\models\train_xgb.py

# Launch dashboard
python -m streamlit run src\dashboard\visualization.py