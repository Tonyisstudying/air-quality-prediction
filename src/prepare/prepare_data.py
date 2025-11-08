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
