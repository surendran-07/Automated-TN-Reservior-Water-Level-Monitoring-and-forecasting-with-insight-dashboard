import os
import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

        # ===== FORECAST =====
        last_exog = exog.iloc[-1:] if not exog.empty else None
        future_exog = (
            pd.concat([last_exog] * FORECAST_DAYS)
            if last_exog is not None else None
        )

        fc = res.get_forecast(steps=FORECAST_DAYS, exog=future_exog)
        preds = fc.predicted_mean
        conf_int = fc.conf_int()

        last_date = y.index[-1]
        prev_level = y.iloc[-1]

        for i in range(FORECAST_DAYS):
            current_pred = preds.iloc[i]

            if current_pred > prev_level:
                trend = "Rising"
            elif current_pred < prev_level:
                trend = "Falling"
            else:
                trend = "Stable"

            all_outputs.append({
                "reservoir_name": reservoir,
                "forecast_date": last_date + timedelta(days=i + 1),
                "forecast_level_ft": round(current_pred, 3),
                "forecast_min_level_ft": round(conf_int.iloc[i, 0], 3),
                "forecast_max_level_ft": round(conf_int.iloc[i, 1], 3),
                "forecast_trend": trend
            })

            prev_level = current_pred

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(reservoir).lower())
        pd.DataFrame(all_outputs).to_csv(
            os.path.join(FORECAST_DIR, f"{safe_name}_forecast.csv"),
            index=False
        )

    pd.DataFrame(all_outputs).to_csv(
        os.path.join(PROCESSED_DIR, "forecast_summary.csv"),
        index=False
    )

    print(" Forecasting with Trend + Min/Max completed successfully")

# =========================
if __name__ == "__main__":
    run_forecasting()
