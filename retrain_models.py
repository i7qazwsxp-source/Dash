"""
retrain_models.py
=================
يعيد تدريب النماذج الثلاثة وحفظها بـ scikit-learn 1.8.0 / Python 3.12
شغّل من نفس المجلد الذي يحتوي على Chicago_Traffic_7days.csv
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
CSV_PATH        = "Chicago_Traffic_7days.csv"
FORECAST_HORIZON = 3
N_LAGS          = 6
RANDOM_SEED     = 42
np.random.seed(RANDOM_SEED)

MASTER_FEATURES = [
    'SEGMENT_CODE', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'WEEKEND',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6',
    'roll_mean_3', 'roll_mean_6', 'roll_std_3'
]

print("="*55)
print("  Chicago Traffic — Model Retraining Script")
print("="*55)

# ─── Load ─────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
assert os.path.exists(CSV_PATH), f"File not found: {CSV_PATH}"
df_raw = pd.read_csv(CSV_PATH)
print(f"      Rows: {len(df_raw):,}")

# ─── Clean ────────────────────────────────────────────────────────────────────
print("[2/6] Cleaning...")
df = df_raw.copy()
df["TIME"]  = pd.to_datetime(df["TIME"], errors="coerce")
df["SPEED"] = pd.to_numeric(df["SPEED"], errors="coerce")
df = df.dropna(subset=["TIME", "SEGMENT_ID", "SPEED"])
df = df[df["SPEED"] > 0].copy()
df = df.sort_values(["SEGMENT_ID", "TIME"]).reset_index(drop=True)

# ─── Features ─────────────────────────────────────────────────────────────────
print("[3/6] Engineering features...")
df["HOUR"]       = pd.to_numeric(df["HOUR"], errors="coerce")
df["MONTH"]      = pd.to_numeric(df["MONTH"], errors="coerce")
df["DAY_OF_WEEK"]= pd.to_numeric(df["DAY_OF_WEEK"], errors="coerce")
df["WEEKEND"]    = df["DAY_OF_WEEK"].isin([1, 7]).astype(int)

g = df.groupby("SEGMENT_ID", sort=False)
for k in range(1, N_LAGS + 1):
    df[f"lag_{k}"] = g["SPEED"].shift(k)

shifted = g["SPEED"].shift(1)
df["roll_mean_3"] = shifted.groupby(df["SEGMENT_ID"]).rolling(3).mean().reset_index(level=0, drop=True)
df["roll_mean_6"] = shifted.groupby(df["SEGMENT_ID"]).rolling(6).mean().reset_index(level=0, drop=True)
df["roll_std_3"]  = shifted.groupby(df["SEGMENT_ID"]).rolling(3).std().reset_index(level=0, drop=True)
df["SPEED_FUTURE"] = g["SPEED"].shift(-FORECAST_HORIZON)

# Encode SEGMENT_ID
segment_codes, _ = pd.factorize(df["SEGMENT_ID"], sort=True)
df["SEGMENT_CODE"] = segment_codes.astype(np.int32)

feature_cols = ["SEGMENT_CODE"] + [c for c in MASTER_FEATURES if c != "SEGMENT_CODE"]
df_model = df.dropna(subset=feature_cols + ["SPEED_FUTURE"]).copy()
print(f"      Modeling rows: {len(df_model):,}")

# ─── Split ────────────────────────────────────────────────────────────────────
print("[4/6] Splitting train/val/test...")
df_model = df_model.sort_values("TIME").reset_index(drop=True)
df_model["DATE"] = df_model["TIME"].dt.date
unique_dates = sorted(df_model["DATE"].unique())

train_dates = unique_dates[:4]
val_dates   = unique_dates[4:5]
test_dates  = unique_dates[5:7]

train_df = df_model[df_model["DATE"].isin(train_dates)]
val_df   = df_model[df_model["DATE"].isin(val_dates)]
test_df  = df_model[df_model["DATE"].isin(test_dates)]

X_train = train_df[MASTER_FEATURES].values
y_train = train_df["SPEED_FUTURE"].values
X_val   = val_df[MASTER_FEATURES].values
y_val   = val_df["SPEED_FUTURE"].values

X_tune  = np.vstack([X_train, X_val])
y_tune  = np.concatenate([y_train, y_val])
ps      = PredefinedSplit([-1]*len(X_train) + [0]*len(X_val))

print(f"      Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(test_df):,}")

# ─── Train ────────────────────────────────────────────────────────────────────
print("[5/6] Training models (this may take a few minutes)...")

models_config = {
    "histgb": {
        "model":  HistGradientBoostingRegressor(random_state=42, early_stopping=False),
        "params": {"learning_rate": [0.01, 0.1], "max_iter": [300, 500], "max_depth": [6, 10]}
    },
    "xgboost": {
        "model":  XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        "params": {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500], "max_depth": [6, 10]}
    },
    "lightgbm": {
        "model":  LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "params": {"learning_rate": [0.01, 0.1], "n_estimators": [300, 500], "max_depth": [6, 10]}
    }
}

for name, config in models_config.items():
    t0 = time.time()
    print(f"\n  -> {name}...")

    search = RandomizedSearchCV(
        estimator=config["model"],
        param_distributions=config["params"],
        n_iter=5,
        scoring="neg_root_mean_squared_error",
        cv=ps,
        refit=False,
        verbose=0,
        random_state=42
    )
    search.fit(pd.DataFrame(X_tune, columns=MASTER_FEATURES), y_tune)

    best_model = config["model"].set_params(**search.best_params_)
    best_model.fit(pd.DataFrame(X_train, columns=MASTER_FEATURES), y_train)

    filename = f"{name}_finetuned.pkl"
    joblib.dump({"model": best_model, "features": MASTER_FEATURES}, filename)
    print(f"     Saved: {filename}  ({time.time()-t0:.0f}s)")

# ─── Thresholds ───────────────────────────────────────────────────────────────
print("\n[6/6] Computing traffic thresholds...")
df_clean = df[df["SPEED"] > 0].copy()
segment_stats = df_clean.groupby("SEGMENT_ID")["SPEED"].agg(
    v15=lambda x: x.quantile(0.15),
    v85=lambda x: x.quantile(0.85)
).to_dict("index")
joblib.dump(segment_stats, "traffic_thresholds.pkl")
print(f"      Saved: traffic_thresholds.pkl ({len(segment_stats)} segments)")

print("\n" + "="*55)
print("  DONE! All models retrained and saved.")
print("  Files to upload to GitHub:")
print("    - histgb_finetuned.pkl")
print("    - xgboost_finetuned.pkl")
print("    - lightgbm_finetuned.pkl")
print("    - traffic_thresholds.pkl")
print("="*55)
