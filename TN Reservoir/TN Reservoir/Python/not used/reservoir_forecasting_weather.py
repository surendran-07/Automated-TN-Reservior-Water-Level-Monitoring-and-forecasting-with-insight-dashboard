import os
import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =========================
# PATHS
# =========================
BASE_DIR = r"C:\Users\Admin\Documents\UiPath\TN Reservoir"
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed")
FORECAST_DIR = os.path.join(PROCESSED_DIR, "forecasts")

INPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_master.csv")
FORECAST_DAYS = 7

os.makedirs(FORECAST_DIR, exist_ok=True)

# =========================
# SAFE METRICS
# =========================
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

# =========================
# COLUMN DETECTION
# =========================
def detect_column(cols, keywords):
    for c in cols:
        for k in keywords:
            if k in c:
                return c
    return None

# =========================
# MAIN PIPELINE
# =========================
def run_forecasting():

    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip().str.lower()

    date_col = detect_column(df.columns, ["date"])
    reservoir_col = detect_column(df.columns, ["reservoir", "dam", "name"])
    level_col = detect_column(df.columns, ["current_level", "water_level", "level"])

    if not all([date_col, reservoir_col, level_col]):
        raise ValueError("Required columns not found (date / reservoir / level)")

    inflow_col = detect_column(df.columns, ["inflow"])
    percent_col = detect_column(df.columns, ["percent"])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    all_outputs = []

    for reservoir, grp in df.groupby(reservoir_col):

        grp = grp.sort_values(date_col).set_index(date_col)

        y = grp[level_col].astype(float)
        y = y.asfreq("D").interpolate()

        if len(y) < 15:
            continue

        exog = pd.DataFrame(index=y.index)

        if inflow_col:
            exog["inflow"] = grp[inflow_col].reindex(y.index).fillna(0)

        if percent_col:
            exog["percent_full"] = grp[percent_col].reindex(y.index).ffill()

        model = SARIMAX(
            y,
            exog=exog if not exog.empty else None,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        res = model.fit(disp=False)

        # ===== Forecast =====
        last_exog = exog.iloc[-1:] if not exog.empty else None
        future_exog = (
            pd.concat([last_exog] * FORECAST_DAYS)
            if last_exog is not None else None
        )

        fc = res.get_forecast(steps=FORECAST_DAYS, exog=future_exog)
        preds = fc.predicted_mean

        # ===== Metrics (BACK-TEST) =====
        test_len = min(FORECAST_DAYS, len(y))
        y_true = y[-test_len:]
        y_pred = res.fittedvalues[-test_len:]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # ✅ FIXED
        mape = safe_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        last_date = y.index[-1]

        for i in range(FORECAST_DAYS):
            all_outputs.append({
                "reservoir_name": reservoir,
                "forecast_date": last_date + timedelta(days=i + 1),
                "forecast_level_ft": round(preds.iloc[i], 3),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4) if not np.isnan(mape) else None,
                "r2_score": round(r2, 4)
            })

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(reservoir).lower())
        pd.DataFrame(all_outputs).to_csv(
            os.path.join(FORECAST_DIR, f"{safe_name}_forecast.csv"),
            index=False
        )

    pd.DataFrame(all_outputs).to_csv(
        os.path.join(PROCESSED_DIR, "forecast_summary.csv"),
        index=False
    )

    print("✅ Forecasting completed successfully")

# =========================
if __name__ == "__main__":
    run_forecasting()
