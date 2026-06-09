import os
import joblib
import numpy as np
import pandas as pd

MASTER_FEATURES = [
    'SEGMENT_CODE', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'WEEKEND',
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6',
    'roll_mean_3', 'roll_mean_6', 'roll_std_3'
]

# 3-model ensemble only — Random Forest excluded per Final Report
ALLOWED_MODELS = ['histgb', 'xgboost', 'lightgbm']

# CI thresholds per Final Report
CI_FREE_FLOW  = 0.90   # CI >= 0.90 → Green
CI_MODERATE   = 0.75   # CI >= 0.75 → Yellow
                      # CI <  0.5  → Red

class TrafficPredictor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.models = {}
        self.v85_lookup = {}
        self.load_artifacts()

    def load_artifacts(self):
        print("\n" + "="*50)
        print("   Chicago Traffic — Inference Engine")
        print("="*50)

        # Load thresholds
        thresh_path = os.path.join(self.base_path, 'traffic_thresholds.pkl')
        if os.path.exists(thresh_path):
            self.v85_lookup = joblib.load(thresh_path)
            print(f"  [OK] traffic_thresholds.pkl    ({len(self.v85_lookup)} segments)")
        else:
            print("  [!!] traffic_thresholds.pkl     NOT FOUND — defaulting V85 to 30.0")

        # Load only the 3 allowed models
        loaded = 0
        for name in ALLOWED_MODELS:
            path = os.path.join(self.base_path, f'{name}_finetuned.pkl')
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"  [OK] {name}_finetuned.pkl")
                loaded += 1
            else:
                print(f"  [!!] {name}_finetuned.pkl       NOT FOUND — skipped")

        print("-"*50)
        if loaded == 3:
            print(f"  Ensemble ready — {loaded}/3 models loaded")
            print("  Open your browser at: http://localhost:8501")
        else:
            print(f"  WARNING: only {loaded}/3 models loaded")
        print("="*50 + "\n")

    def predict_speed(self, input_df: pd.DataFrame) -> np.ndarray:
        print("\n--- LOG: Running Ensemble Prediction ---")

        # Schema contract enforcement
        missing = [f for f in MASTER_FEATURES if f not in input_df.columns]
        if missing:
            raise ValueError(f"Schema violation — missing features: {missing}")

        X = input_df[MASTER_FEATURES].copy()
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        if not self.models:
            print("WARNING: No models loaded. Returning zeros.")
            return np.zeros(len(input_df))

        preds = []
        for name, artifact in self.models.items():
            model = artifact['model'] if isinstance(artifact, dict) else artifact
            preds.append(model.predict(X))
            print(f"LOG: Predictions from -> {name}")

        final_pred = np.mean(preds, axis=0)
        print("LOG: Ensemble complete.")
        return final_pred

    def apply_ci_colors(self, df: pd.DataFrame) -> pd.DataFrame:
        # Map V85 per segment — fallback to 30.0 if not found or zero
        df['v85'] = df['SEGMENT_ID'].map(
            lambda x: self.v85_lookup.get(int(float(x)), {}).get('v85', 30.0)
        )
        df['v85'] = np.where((df['v85'] <= 0) | df['v85'].isna(), 30.0, df['v85'])

        # CI = V_pred / V85  (capped between 0 and 1)
        df['CI'] = (df['SPEED'] / df['v85']).clip(0, 1)

        # Color mapping per Final Report
        conditions  = [df['CI'] >= CI_FREE_FLOW, df['CI'] >= CI_MODERATE]
        color_vals  = ['#2d9e4f', '#f4a823']
        label_vals  = ['Free Flow', 'Moderate']

        df['TRAFFIC_COLOR'] = np.select(conditions, color_vals, default='#e63946')
        df['TRAFFIC_LABEL'] = np.select(conditions, label_vals, default='Heavy Congestion')

        return df
