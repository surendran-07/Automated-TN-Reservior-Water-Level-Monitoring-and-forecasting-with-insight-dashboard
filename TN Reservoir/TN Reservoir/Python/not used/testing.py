#!/usr/bin/env python3
"""
reservoir_pipeline_final.py
STABLE + EXTENDED VERSION
"""

import os
import re
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ================= CONFIG =================
BASE = r"C:\Users\Admin\Documents\UiPath\TN Reservoir"
DATA_DIR = os.path.join(BASE, "Data")
OUT_DIR = os.path.join(BASE, "Processed")

MASTER_FILE = os.path.join(DATA_DIR, "master.xlsx")

CLEANED_MASTER_CSV = os.path.join(OUT_DIR, "cleaned_master.csv")
CLEANED_MASTER_XLSX = os.path.join(OUT_DIR, "cleaned_master.xlsx")
FORECAST_COMBINED = os.path.join(OUT_DIR, "forecasts_combined.csv")
INSIGHT_COMBINED = os.path.join(OUT_DIR, "insights_combined.csv")

os.makedirs(OUT_DIR, exist_ok=True)

FORECAST_DAYS = 7

# ================= HELPERS =================
def detect_column(df, keys):
    for c in df.columns:
        s = str(c).lower().replace("_", "").replace(" ", "")
        if any(k in s for k in keys):
            return c
    return None

def classify_trend(val, threshold=0.5):
    if pd.isna(val):
        return "Stable"
    if val > threshold:
        return "Rising"
    if val < -threshold:
        return "Falling"
    return "Stable"

# ================= NORMALIZE =================
def normalize_master_df(df):
    out = pd.DataFrame()

    # REQUIRED (must exist)
    date_col = detect_column(df, ["date"])
    name_col = detect_column(df, ["reservoir", "dam", "name"])
    level_col = detect_column(df, ["level"])

    if not all([date_col, name_col, level_col]):
        raise ValueError("Master file must contain Date, Reservoir Name, and Level columns")

    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    out["reservoir_name"] = df[name_col].astype(str)
    out["current_level_ft"] = pd.to_numeric(df[level_col], errors="coerce")

    # OPTIONAL columns (safe)
    storage_col = detect_column(df, ["storage"])
    percent_col = detect_column(df, ["percent"])

    out["storage_m3"] = (
        pd.to_numeric(df[storage_col], errors="coerce")
        if storage_col else np.nan
    )

    out["percent_full"] = (
        pd.to_numeric(
            df[percent_col].astype(str).str.replace("%", ""),
            errors="coerce"
        )
        if percent_col else np.nan
    )

    return out


# ================= CLEANED MASTER =================
def build_cleaned_master(df):

    df = df.sort_values(["reservoir_name", "date"]).reset_index(drop=True)

    df["previous_day_level_ft"] = df.groupby("reservoir_name")["current_level_ft"].shift(1)
    df["level_change_ft"] = df["current_level_ft"] - df["previous_day_level_ft"]
    df["percent_change_from_previous_day"] = (
        df["level_change_ft"] / df["previous_day_level_ft"] * 100
    )

    df["water_trend_1d"] = df["level_change_ft"].apply(classify_trend)

    df["avg_level_7d"] = (
        df.groupby("reservoir_name")["current_level_ft"]
        .rolling(7, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["water_trend_7d"] = (
        df.groupby("reservoir_name")["current_level_ft"]
        .rolling(7, min_periods=2)
        .apply(lambda x: x.iloc[-1] - x.iloc[0])
        .reset_index(level=0, drop=True)
        .apply(classify_trend)
    )

    return df.round(2)

# ================= FORECAST (UNCHANGED MODEL) =================
def sarimax_forecast(series):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,0,1,7))
    res = model.fit(disp=False)

    fc = res.get_forecast(steps=FORECAST_DAYS).summary_frame()
    fc = fc.reset_index().rename(columns={
        "index": "date",
        "mean": "yhat",
        "mean_ci_lower": "yhat_lower",
        "mean_ci_upper": "yhat_upper"
    })

    return fc[["date", "yhat", "yhat_lower", "yhat_upper"]]

# ================= INSIGHTS =================
def build_insights(cleaned, forecasts):

    rows = []

    for r in cleaned["reservoir_name"].unique():
        hist = cleaned[cleaned["reservoir_name"] == r].sort_values("date")
        latest = hist.iloc[-1]

        fc = forecasts[forecasts["reservoir_name"] == r].sort_values("date")
        if not fc.empty:
            fc_trend = classify_trend(fc["yhat"].iloc[-1] - fc["yhat"].iloc[0])
        else:
            fc_trend = "Stable"

        rows.append({
            "reservoir_name": r,
            "latest_date": latest["date"],
            "latest_level_ft": latest["current_level_ft"],
            "previous_day_level_ft": latest["previous_day_level_ft"],
            "level_change_ft": latest["level_change_ft"],
            "percent_change_from_previous_day": latest["percent_change_from_previous_day"],
            "water_trend_1d": latest["water_trend_1d"],
            "water_trend_7d": latest["water_trend_7d"],
            "forecast_direction_trend": fc_trend
        })

    return pd.DataFrame(rows)

# ================= MAIN =================
def run_pipeline():

    master = pd.read_excel(MASTER_FILE)
    norm = normalize_master_df(master)
    cleaned = build_cleaned_master(norm)

    cleaned.to_csv(CLEANED_MASTER_CSV, index=False)
    cleaned.to_excel(CLEANED_MASTER_XLSX, index=False)

    forecasts = []

    for r in cleaned["reservoir_name"].unique():
        sub = cleaned[cleaned["reservoir_name"] == r].copy()
        series = sub.set_index(pd.to_datetime(sub["date"]))["current_level_ft"].dropna()

        if len(series) < 10:
            continue

        fc = sarimax_forecast(series)
        fc["reservoir_name"] = r
        forecasts.append(fc)

    forecast_df = pd.concat(forecasts, ignore_index=True)
    forecast_df.to_csv(FORECAST_COMBINED, index=False)

    insights = build_insights(cleaned, forecast_df)
    insights.to_csv(INSIGHT_COMBINED, index=False)

    print("Pipeline completed successfully")

if __name__ == "__main__":
    run_pipeline()
