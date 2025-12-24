import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# =========================
# PATHS (SAME AS FORECAST FILE)
# =========================
BASE_DIR = r"C:\Users\Admin\Documents\UiPath\TN Reservoir"
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed")

INPUT_FILE = os.path.join(PROCESSED_DIR, "cleaned_master.csv")
ACCURACY_FILE = os.path.join(PROCESSED_DIR, "model_accuracy_summary.csv")

FORECAST_DAYS = 7

# =========================
# SAFE MAPE
# =========================
def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# =========================
# COLUMN DETECTION (SAME LOGIC)
# =========================
def detect_column(cols, keywords):
    for c in cols:
        for k in keywords:
            if k in c:
                return c
    return None

# =========================
# MAIN ACCURACY PIPELINE
# =========================
def run_accuracy():

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

    accuracy_rows = []

    # =========================
    # PER RESERVOIR ACCURACY
    # =========================
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

        # ===== BACK-TEST METRICS =====
        test_len = min(FORECAST_DAYS, len(y))
        y_true = y[-test_len:]
        y_pred = res.fittedvalues[-test_len:]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = safe_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        accuracy_percent = None
        if not np.isnan(mape):
            accuracy_percent = max(0, 100 - mape)

        accuracy_rows.append({
            "reservoir_name": reservoir,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape_percent": round(mape, 2) if not np.isnan(mape) else None,
            "r2_score": round(r2, 4),
            "model_accuracy_percent": round(accuracy_percent, 2) if accuracy_percent else None
        })

    # =========================
    # SAVE ACCURACY FILE
    # =========================
    pd.DataFrame(accuracy_rows).to_csv(ACCURACY_FILE, index=False)

    print("✅ Model accuracy file generated successfully")


# =========================
if __name__ == "__main__":
    run_accuracy()
