import pandas as pd
import glob
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

csv_files = glob.glob('data/raw/*.csv')

#Mapping schema to a common format
columns_mapping= {
    'beijing_pm25.csv': {
        'datetime':'datetime',
        'pm25':'PM2.5',
        'station':'station_id'
    },
    'beijing_pm10.csv': {
        'datetime':'datetime',
        'pm10':'PM10',
        'station':'station_id'
    },
    'beijing_weather.csv': {
        'dt':'datetime',
        'temp':'temperature',
        'humidity':'humidity',
        'wind_speed':'wind_speed'
    },
    'openaq_beijing.csv': {
        'date':'datetime',
        'value':'PM2.5',
        'location':'station_id'
    },
    'station_metadata.csv': {
        'station':'station_id',
        'lat':'latitude',
        'lon':'longitude',
        'elevation':'elevation'
    }
}

dfs = {}
for file in csv_files:
    filename = os.path.basename(file)
    df = pd.read_csv(file)
    if filename in columns_mapping:
        df = df.rename(columns=columns_mapping[filename])
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    dfs[filename] = df

# Merging datasets on datetime and station_id
merged_df = dfs['beijing_pm25.csv']

for key in ['beijing_pm10.csv', 'beijing_weather.csv', 'openaq_beijing.csv']:
    merged_df = pd.merge(merged_df, dfs[key], on=['datetime', 'station_id'], how='outer')

# Merging station metadata
merged_df = pd.merge(merged_df, dfs['station_metadata.csv'], on='station_id', how='left')

# Handling missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(merged_df.select_dtypes(include=['int64', 'float64']))
imputed_df = pd.DataFrame(imputed_data, columns=merged_df.select_dtypes(include=['int64','float64']).columns)
for col in imputed_df.columns:
    merged_df[col] = imputed_df[col]

# Standardizing numerical features
scaler = StandardScaler()
numerical_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
merged_df[numerical_cols] = scaler.fit_transform(merged_df[numerical_cols])

# Drop rows with missing datetime or station_id values
merged_df = merged_df.dropna(subset=['datetime', 'station_id'])

# Saving the cleaned and processed data
os.makedirs('data/prepared', exist_ok=True)
merged_df.to_csv('data/prepared/cleaned_beijing_data.csv', index=False)

