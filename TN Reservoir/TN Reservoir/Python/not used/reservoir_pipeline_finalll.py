#!/usr/bin/env python3
# ==========================================================
# TN RESERVOIR – FINAL PIPELINE (ALL-DAY WATER TREND)
# ==========================================================

import os
import re
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ================= CONFIG =================
BASE_DIR = Path(r"C:\Users\Admin\Documents\UiPath\TN Reservoir")
DATA_DIR = BASE_DIR / "Data"
PROCESSED_DIR = BASE_DIR / "Processed"
FORECAST_DIR = PROCESSED_DIR / "forecasts"
INSIGHT_DIR = PROCESSED_DIR / "insights"

MASTER_FILE = DATA_DIR / "master.xlsx"
FORECAST_DAYS = 7

PROCESSED_DIR.mkdir(exist_ok=True)
FORECAST_DIR.mkdir(exist_ok=True)
INSIGHT_DIR.mkdir(exist_ok=True)

# ================= CONSTANTS =================
M3_PER_MCFT = 28_316.846592
CUSEC_TO_M3_PER_DAY = 0.028316846592 * 86400

# ================= HELPERS =================
def detect_column(df, keys):
    for c in df.columns:
        n = str(c).lower().replace(" ", "").replace("_", "")
        for k in keys:
            if k in n:
                return c
    return None

def parse_num(v):
    if pd.isna(v):
        return np.nan
    try:
        return float(re.sub(r"[^\d\.\-]", "", str(v)))
    except:
        return np.nan

def safe_name(s):
    return re.sub(r'[\\/*?:"<>|]', "_", str(s))

# ================= NORMALIZE MASTER =================
def normalize_master(df):
    out = pd.DataFrame()

    out["date"] = pd.to_datetime(
        df[detect_column(df, ["date"])], errors="coerce"
    ).dt.date

    out["reservoir_name"] = df[
        detect_column(df, ["reservoir","dam","name"])
    ].astype(str)

    out["current_level_ft"] = pd.to_numeric(
        df[detect_column(df, ["current","level"])], errors="coerce"
    )

    out["storage_m3"] = (
        df[detect_column(df, ["storage"])]
        .apply(parse_num) * M3_PER_MCFT
    )

    out["inflow_m3_per_day"] = (
        df[detect_column(df, ["inflow"])]
        .apply(parse_num) * CUSEC_TO_M3_PER_DAY
    )

    out["outflow_m3_per_day"] = (
        df[detect_column(df, ["outflow"])]
        .apply(parse_num) * CUSEC_TO_M3_PER_DAY
    )

    out["full_capacity_m3"] = (
        df[detect_column(df, ["capacity"])]
        .apply(parse_num) * M3_PER_MCFT
    )

    pct = detect_column(df, ["percent"])
    out["percent_full"] = (
        pd.to_numeric(df[pct].astype(str).str.replace("%",""), errors="coerce")
        if pct else np.nan
    )

    return out

# ================= 7-DAY TREND (ALL DATES) =================
def compute_7day_trend(group):
    group = group.sort_values("date").reset_index(drop=True)
    trends = []

    for i in range(len(group)):
        if i < 6:
            trends.append("Insufficient Data")
        else:
            delta = (
                group.loc[i, "current_level_ft"]
                - group.loc[i - 6, "current_level_ft"]
            )
            if delta > 0.5:
                trends.append("Rising")
            elif delta < -0.5:
                trends.append("Falling")
            else:
                trends.append("Stable")

    group["water_trend_7d"] = trends
    return group

# ================= CLEANED MASTER =================
def build_cleaned_master(df):
    df = df.sort_values(["reservoir_name","date"]).reset_index(drop=True)

    # Fill percent_full
    df["percent_full"] = df.apply(
        lambda r:
        r["percent_full"]
        if pd.notna(r["percent_full"])
        else (r["storage_m3"] / r["full_capacity_m3"] * 100)
        if pd.notna(r["storage_m3"]) and pd.notna(r["full_capacity_m3"])
        else np.nan,
        axis=1
    )

    # Previous day & percent change
    df["previous_day_level"] = df.groupby("reservoir_name")["current_level_ft"].shift(1)
    df["percent_change"] = (
        (df["current_level_ft"] - df["previous_day_level"])
        / df["previous_day_level"] * 100
    )

    # Water availability
    df["water_available_days"] = df.apply(
        lambda r:
        r["storage_m3"] / r["outflow_m3_per_day"]
        if pd.notna(r["storage_m3"]) and pd.notna(r["outflow_m3_per_day"])
        and r["outflow_m3_per_day"] > 0
        else np.nan,
        axis=1
    )

    # Low water warning
    df["water_low_warning"] = df["percent_full"] < 20

    # Apply 7-day trend for ALL dates
    df = (
        df.groupby("reservoir_name", group_keys=False)
          .apply(compute_7day_trend)
    )

    return df

# ================= FORECAST =================
def sarimax_forecast(series):
    if len(series) < 10:
        last = series.iloc[-1]
        idx = pd.date_range(
            series.index[-1] + pd.Timedelta(days=1),
            periods=FORECAST_DAYS
        )
        return pd.DataFrame({
            "date": idx,
            "yhat": last,
            "yhat_lower": last,
            "yhat_upper": last
        })

    model = SARIMAX(
        series,
        order=(1,1,1),
        seasonal_order=(1,0,1,7)
    )
    res = model.fit(disp=False)
    f = res.get_forecast(FORECAST_DAYS).summary_frame().reset_index()

    return f.rename(columns={
        "index":"date",
        "mean":"yhat",
        "mean_ci_lower":"yhat_lower",
        "mean_ci_upper":"yhat_upper"
    })

# ================= INSIGHTS (LATEST DAY) =================
def build_insights(cleaned):
    rows = []
    for r in cleaned["reservoir_name"].unique():
        latest = cleaned[cleaned["reservoir_name"] == r].iloc[-1]
        rows.append({
            "reservoir_name": r,
            "latest_date": latest["date"],
            "latest_level_ft": latest["current_level_ft"],
            "previous_day_level": latest["previous_day_level"],
            "percent_change": latest["percent_change"],
            "percent_full": latest["percent_full"],
            "water_trend_7d": latest["water_trend_7d"],
            "water_available_days": latest["water_available_days"],
            "water_low_warning": latest["water_low_warning"]
        })
    return pd.DataFrame(rows)

# ================= MAIN =================
def run_pipeline():
    print("Pipeline started:", datetime.now())

    master = pd.read_excel(MASTER_FILE)
    norm = normalize_master(master)
    cleaned = build_cleaned_master(norm)

    cleaned.to_csv(PROCESSED_DIR / "cleaned_master.csv", index=False)
    cleaned.to_excel(PROCESSED_DIR / "cleaned_master.xlsx", index=False)

    all_fc = []
    for r in cleaned["reservoir_name"].unique():
        s = cleaned[cleaned["reservoir_name"] == r]
        series = s.set_index(pd.to_datetime(s["date"]))["current_level_ft"].dropna()
        if series.empty:
            continue
        fc = sarimax_forecast(series)
        fc["reservoir_name"] = r
        fc.to_csv(FORECAST_DIR / f"{safe_name(r)}_forecast.csv", index=False)
        all_fc.append(fc)

    if all_fc:
        pd.concat(all_fc).to_csv(PROCESSED_DIR / "forecasts_combined.csv", index=False)

    insights = build_insights(cleaned)
    insights.to_csv(PROCESSED_DIR / "insights_combined.csv", index=False)

    for _, row in insights.iterrows():
        pd.DataFrame([row]).to_csv(
            INSIGHT_DIR / f"{safe_name(row['reservoir_name'])}_insights.csv",
            index=False
        )

    print("Pipeline finished successfully")

# ================= RUN =================
if __name__ == "__main__":
    run_pipeline()
