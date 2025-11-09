python src\prepare\prepare_data.py
python src\models\train_lstm.py
python src\models\train_xgb.py
python -m streamlit run src\dashboard\visualization.py