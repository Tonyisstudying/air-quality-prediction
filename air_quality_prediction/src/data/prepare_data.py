import os
import glob
import pandas as pd
from sklearn.impute import KNNImputer


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        return df

    year_cols = [c for c in df.columns if c in {'year', 'yr'}]
    month_cols = [c for c in df.columns if c in {'month', 'mon'}]
    day_cols = [c for c in df.columns if c in {'day', 'dy', 'date'}]
    hour_cols = [c for c in df.columns if c in {'hour', 'hr'}]

    if year_cols and month_cols and day_cols and hour_cols:
        y = df[year_cols[0]].astype('Int64')
        m = df[month_cols[0]].astype('Int64')
        d = df[day_cols[0]].astype('Int64')
        h = df[hour_cols[0]].astype('Int64')
        # Fallbacks to 0 for missing components before to_datetime
        y = y.fillna(1970)
        m = m.fillna(1)
        d = d.fillna(1)
        h = h.fillna(0)
        df['datetime'] = pd.to_datetime(
            {
                'year': y.astype(int),
                'month': m.astype(int),
                'day': d.astype(int),
                'hour': h.astype(int),
            },
            errors='coerce',
        )
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # No clear datetime components; leave as-is and handle later
        df['datetime'] = pd.NaT

    return df


def _ensure_station(df: pd.DataFrame, source_filename: str) -> pd.DataFrame:
    df = df.copy()
    if 'station' in df.columns:
        return df

    # Derive station from PRSA filename pattern if possible
    base = os.path.basename(source_filename)
    station_name = None
    if base.startswith('PRSA_Data_') and '_' in base:
        try:
            # Example: PRSA_Data_Aotizhongxin_20130301-20170228.csv
            station_name = base.split('PRSA_Data_')[1].split('_')[0]
        except Exception:
            station_name = None

    if station_name is None:
        station_name = os.path.splitext(base)[0]

    df['station'] = station_name
    return df


def load_and_fuse_raw(raw_dir: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin1')

        df = _standardize_columns(df)
        df = _ensure_datetime(df)
        df = _ensure_station(df, path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f'No CSV files found in {raw_dir}')

    # Union of columns; concatenate and align by columns
    fused = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Drop rows with no datetime
    fused = fused.dropna(subset=['datetime'])

    # Ensure consistent dtypes where reasonable
    # Lowercase station names for consistency
    fused['station'] = fused['station'].astype(str).str.strip()

    # Remove exact duplicates by datetime+station if both exist
    if {'datetime', 'station'}.issubset(fused.columns):
        fused = fused.sort_values(['station', 'datetime']).drop_duplicates(
            subset=['station', 'datetime'], keep='first'
        )

    fused = fused.reset_index(drop=True)
    return fused


def impute_numeric_knn(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    df = df.copy()

    # Select numeric columns for imputation; keep identifiers intact
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        return df

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(df[numeric_cols])
    df[numeric_cols] = imputed
    return df


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = os.path.join(project_root, 'data', 'raw')
    out_dir = os.path.join(project_root, 'data', 'prepared')
    os.makedirs(out_dir, exist_ok=True)

    fused = load_and_fuse_raw(raw_dir)
    fused_imputed = impute_numeric_knn(fused, n_neighbors=5)

    out_path = os.path.join(out_dir, 'fused_imputed.csv')
    fused_imputed.to_csv(out_path, index=False)

    print(f'Saved prepared data to {out_path}')


if __name__ == '__main__':
    main()

import pandas as pd
import glob
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

csv_files = glob.glob('air_quality_prediction/data/raw/*.csv')

# Mapping schema to a common format
columns_mapping = {
    'beijing_pm25.csv': {
        'datetime': 'datetime',
        'pm25': 'PM2.5',
        'station': 'station_id'
    },
    'beijing_pm10.csv': {
        'datetime': 'datetime',
        'pm10': 'PM10',
        'station': 'station_id'
    },
    'beijing_weather.csv': {
        'dt': 'datetime',
        'temp': 'temperature',
        'humidity': 'humidity',
        'wind_speed': 'wind_speed'
    },
    'openaq_beijing.csv': {
        'date': 'datetime',
        'value': 'PM2.5',
        'location': 'station_id'
    },
    'station_metadata.csv': {
        'station': 'station_id',
        'lat': 'latitude',
        'lon': 'longitude',
        'elevation': 'elevation'
    }
}

# Step 1: Create an empty dictionary to store the DataFrames
dfs = {}

# Step 2: Loop over all the CSV files and read them into the dictionary
for file in csv_files:
    filename = os.path.basename(file)
    df = pd.read_csv(file)
    
    # Apply columns mapping if the file is in the mapping
    if filename in columns_mapping:
        df = df.rename(columns=columns_mapping[filename])

    # Convert 'datetime' column to datetime type if it exists
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    
    # Store the DataFrame in the dictionary
    dfs[filename] = df

# Step 3: Merge all dataframes based on 'datetime' and 'station_id'
# Start with the first dataframe in the list, and merge others one by one
merged_df = list(dfs.values())[0]  # Start with the first dataframe in dfs

for key in list(dfs.keys())[1:]:
    merged_df = pd.merge(merged_df, dfs[key], on=['datetime', 'station_id'], how='outer')

# Step 4: Handle missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
imputed_data = imputer.fit_transform(merged_df.select_dtypes(include=['int64', 'float64']))
imputed_df = pd.DataFrame(imputed_data, columns=merged_df.select_dtypes(include=['int64', 'float64']).columns)
for col in imputed_df.columns:
    merged_df[col] = imputed_df[col]

# Step 5: Standardize numerical features
scaler = StandardScaler()
numerical_cols = merged_df.select_dtypes(include=['int64', 'float64']).columns
merged_df[numerical_cols] = scaler.fit_transform(merged_df[numerical_cols])

# Step 6: Drop rows with missing 'datetime' or 'station_id' values
merged_df = merged_df.dropna(subset=['datetime', 'station_id'])

# Step 7: Save the cleaned data to a new directory
os.makedirs('data/prepared', exist_ok=True)
merged_df.to_csv('data/prepared/cleaned_beijing_data.csv', index=False)

print("✅ Data cleaned and saved to data/prepared/cleaned_beijing_data.csv")
