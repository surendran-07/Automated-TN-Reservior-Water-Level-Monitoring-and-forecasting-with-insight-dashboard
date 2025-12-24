#!/usr/bin/env python3
"""
reservoir_pipeline_final.py

Final pipeline (unit conversions to m³ / m³-per-day).
"""

import os
import re
import glob
import json
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
BASE = r"C:\Users\Admin\Documents\UiPath\TN Reservoir"
DATA_DIR = os.path.join(BASE, "Data")
OUT_DIR = os.path.join(BASE, "Processed")
FORECAST_DIR = os.path.join(OUT_DIR, "forecasts")
INSIGHT_DIR = os.path.join(OUT_DIR, "insights")

MASTER_FILE = os.path.join(DATA_DIR, "master.xlsx")
FORECAST_DAYS = 7

CLEANED_MASTER_CSV = os.path.join(OUT_DIR, "cleaned_master.csv")
CLEANED_MASTER_XLSX = os.path.join(OUT_DIR, "cleaned_master.xlsx")
FORECAST_COMBINED = os.path.join(OUT_DIR, "forecasts_combined.csv")
INSIGHT_COMBINED = os.path.join(OUT_DIR, "insights_combined.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "summary.json")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)
os.makedirs(INSIGHT_DIR, exist_ok=True)

# ---------------- HELPERS ----------------
def sanitize_filename(name: str) -> str:
    if name is None:
        return "unknown"
    s = re.sub(r'[\\/*?:"<>|]', "_", str(name))
    s = re.sub(r'\s+', '_', s).strip('_')
    return s[:200]

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    df.to_csv(path, index=False)

def detect_column(df, keywords):
    if df is None or df.empty:
        return None
    for col in df.columns:
        norm = str(col).lower().replace(" ", "").replace("_","")
        for k in keywords:
            if k in norm:
                return col
    return None

# ---------------- Unit conversion helpers ----------------
# 1 MCFT (M.Cft) = 1,000,000 ft³. 1 ft³ = 0.028316846592 m³
M3_PER_MCFT = 1e6 * 0.028316846592       # = 28,316.846592 m3 per MCFT
M3_PER_TMC = 1e9 * 0.028316846592        # in case TMC appears
# 1 cubic foot per second (cusec) = 0.028316846592 m³/s -> *86400 = m³/day
CUSEC_TO_M3_PER_DAY = 0.028316846592 * 86400.0

def parse_storage_to_m3(val):
    if pd.isna(val):
        return np.nan
    s = str(val).upper().replace(",", "").strip()
    if s == "":
        return np.nan
    try:
        # detect TMC / MCFT / numeric
        if "TMC" in s:
            n = float(re.sub(r"[^\d\.\-]","", s.replace("TMC","")))
            return n * M3_PER_TMC
        if "MCFT" in s or "M.CFT" in s or "MCF" in s:
            n = float(re.sub(r"[^\d\.\-]","", s.replace("MCFT","").replace("M.CFT","").replace("MCF","")))
            return n * M3_PER_MCFT
        if "M3" in s or "CUM" in s:
            n = float(re.sub(r"[^\d\.\-]","", s))
            return n
        # numeric heuristics
        v = float(re.sub(r"[^\d\.\-]","", s))
        # if large, treat as m3
        if v > 1e6:
            return v
        # else treat as MCFT (best-effort)
        if 1 <= v <= 1e6:
            return v * M3_PER_MCFT
        return v
    except Exception:
        return np.nan

def parse_numeric(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val).replace(",", "").strip()
        return float(re.sub(r"[^\d\.\-]", "", s))
    except:
        return np.nan

# ---------------- Normalization of master's columns ----------------
def normalize_master_df(df):
    """
    Map the master.xlsx (which matches your screenshot) to canonical columns:
    date, reservoir_name, current_level_ft, storage_m3, inflow_m3_per_day, outflow_m3_per_day, full_capacity_m3, percent_full (may be NaN)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "date","reservoir_name","current_level_ft","storage_m3","inflow_m3_per_day","outflow_m3_per_day","full_capacity_m3","percent_full"
        ])
    df = df.copy()

    # detect columns (based on your daily/master headers)
    col_res = detect_column(df, ["reservoir","reservoirs","name","dam"])
    col_full_depth = detect_column(df, ["fulldepth","full_depth","fulldepth(feet)"])
    col_full_capacity = detect_column(df, ["fullcapacity","full_capacity","full_capacity(m.cft.)","full_capacity_mcft","fullcapacity(m.cft)"])
    col_current_level = detect_column(df, ["currentyearlevel","currentyearlevel(feet)","currentlevel","current_year_level","current_level"])
    col_current_storage = detect_column(df, ["currentyearstorage","current_year_storage","currentyearstorage(m.cft.)","currentyearstorage"])
    col_inflow = detect_column(df, ["currentyearinflow","inflow","inflow(cusecs)","inflow_cusec","currentinflow"])
    col_outflow = detect_column(df, ["currentyearoutflow","outflow","outflow(cusecs)","outflow_cusec","currentoutflow"])
    col_date = detect_column(df, ["date","ds","reportdate"])
    col_percent = detect_column(df, ["percentfull","percent_full","percent","%full","percentage"])

    out = pd.DataFrame()
    # date -> parse to datetime.date (no time)
    if col_date:
        out["date"] = pd.to_datetime(df[col_date], errors="coerce").dt.date
    else:
        out["date"] = pd.to_datetime(df.index).date

    out["reservoir_name"] = df[col_res].astype(str).str.strip() if col_res else df.index.astype(str)

    # current level feet
    if col_current_level:
        out["current_level_ft"] = pd.to_numeric(df[col_current_level].astype(str).str.replace(",",""), errors="coerce")
    else:
        out["current_level_ft"] = np.nan

    # storage (source is in M.Cft. per screenshot) -> convert to m3
    if col_current_storage:
        out["storage_m3"] = df[col_current_storage].apply(parse_storage_to_m3)
    else:
        out["storage_m3"] = np.nan

    # inflow/outflow (cusec -> m3/day)
    if col_inflow:
        out["inflow_m3_per_day"] = df[col_inflow].apply(parse_numeric).apply(lambda x: x * CUSEC_TO_M3_PER_DAY if pd.notna(x) else np.nan)
    else:
        out["inflow_m3_per_day"] = np.nan

    if col_outflow:
        out["outflow_m3_per_day"] = df[col_outflow].apply(parse_numeric).apply(lambda x: x * CUSEC_TO_M3_PER_DAY if pd.notna(x) else np.nan)
    else:
        out["outflow_m3_per_day"] = np.nan

    # full capacity -> convert to m3
    if col_full_capacity:
        out["full_capacity_m3"] = df[col_full_capacity].apply(parse_storage_to_m3)
    else:
        out["full_capacity_m3"] = np.nan

    # percent_full if present
    if col_percent:
        out["percent_full"] = pd.to_numeric(df[col_percent].astype(str).str.replace("%","").str.replace(",",""), errors="coerce")
    else:
        out["percent_full"] = np.nan

    return out[["date","reservoir_name","current_level_ft","storage_m3","inflow_m3_per_day","outflow_m3_per_day","full_capacity_m3","percent_full"]]

# ---------------- Build CLEANED MASTER with all dates ----------------
def build_cleaned_master(master_norm):
    # master_norm contains all historical rows (UiPath-maintained master.xlsx)
    df = master_norm.copy()
    # fill storage_m3 if missing using percent & capacity
    def fill_storage(r):
        s = r.get("storage_m3")
        if pd.notna(s):
            return s
        p = r.get("percent_full")
        cap = r.get("full_capacity_m3")
        if pd.notna(p) and pd.notna(cap) and cap != 0:
            return float(p) / 100.0 * float(cap)
        return np.nan
    df["storage_m3"] = df.apply(fill_storage, axis=1)
    # fill percent if missing via storage / capacity
    def fill_percent(r):
        p = r.get("percent_full")
        if pd.notna(p):
            return p
        s = r.get("storage_m3")
        cap = r.get("full_capacity_m3")
        if pd.notna(s) and pd.notna(cap) and cap != 0:
            return float(s) / float(cap) * 100.0
        return np.nan
    df["percent_full"] = df.apply(fill_percent, axis=1)
    # ensure date is date object
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    # sort and compute deltas per reservoir
    df = df.sort_values(["reservoir_name","date"]).reset_index(drop=True)
    df["delta_level_ft"] = df.groupby("reservoir_name")["current_level_ft"].diff().fillna(0)
    df["delta_storage_m3"] = df.groupby("reservoir_name")["storage_m3"].diff().fillna(0)
    # risk_category based on percent_full
    def risk_from_pct(p):
        try:
            if pd.isna(p): return "UNKNOWN"
            p = float(p)
            if p < 25:
                return "CRITICAL"
            elif p < 40:
                return "LOW"
            else:
                return "NORMAL"
        except:
            return "UNKNOWN"
    df["risk_category"] = df["percent_full"].apply(risk_from_pct)
    df["warning_low"] = df["percent_full"].apply(lambda x: True if (pd.notna(x) and float(x) < 25) else False)
    # reorder columns and rounding
    out = df[["date","reservoir_name","current_level_ft","storage_m3","inflow_m3_per_day","outflow_m3_per_day","full_capacity_m3","percent_full","risk_category","warning_low","delta_level_ft","delta_storage_m3"]].copy()
    # rounding similar to your sample
    out["current_level_ft"] = out["current_level_ft"].round(2)
    out["storage_m3"] = out["storage_m3"].round(2)
    out["inflow_m3_per_day"] = out["inflow_m3_per_day"].apply(lambda v: round(v,2) if pd.notna(v) else v)
    out["outflow_m3_per_day"] = out["outflow_m3_per_day"].apply(lambda v: round(v,2) if pd.notna(v) else v)
    out["full_capacity_m3"] = out["full_capacity_m3"].round(2)
    out["percent_full"] = out["percent_full"].apply(lambda v: round(float(v),8) if pd.notna(v) else v)
    out["delta_level_ft"] = out["delta_level_ft"].round(2)
    out["delta_storage_m3"] = out["delta_storage_m3"].round(2)
    return out

# ---------------- SARIMAX forecasting ----------------
def sarimax_forecast(series, periods=FORECAST_DAYS, order=(1,1,1), seasonal_order=(1,0,1,7)):
    s = series.dropna()
    if s.empty:
        return None
    if len(s) < 10:
        last = float(s.iloc[-1])
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
        return pd.DataFrame({"ds": idx, "yhat":[last]*periods, "yhat_lower":[last]*periods, "yhat_upper":[last]*periods})
    try:
        model = SARIMAX(s, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=periods)
        dfp = pred.summary_frame(alpha=0.05).reset_index().rename(columns={"index":"ds","mean":"yhat","mean_ci_lower":"yhat_lower","mean_ci_upper":"yhat_upper"})
        dfp["ds"] = pd.to_datetime(dfp["ds"])
        return dfp[["ds","yhat","yhat_lower","yhat_upper"]]
    except Exception:
        last = float(s.iloc[-1])
        idx = pd.date_range(start=s.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")
        return pd.DataFrame({"ds": idx, "yhat":[last]*periods, "yhat_lower":[last]*periods, "yhat_upper":[last]*periods})

# ---------------- Build insights (latest day) ----------------
def build_insights(cleaned_master, forecasts_map):
    rows = []
    for r in sorted(cleaned_master["reservoir_name"].dropna().unique()):
        sub = cleaned_master[cleaned_master["reservoir_name"]==r].sort_values("date")
        if sub.empty:
            continue
        latest = sub.iloc[-1]
        prev = sub.iloc[-2] if len(sub) >= 2 else None
        prev_level = prev["current_level_ft"] if (prev is not None and not pd.isna(prev["current_level_ft"])) else None
        latest_level = latest["current_level_ft"] if not pd.isna(latest["current_level_ft"]) else None
        change_percent = None
        if prev_level is not None and prev_level != 0 and latest_level is not None:
            change_percent = (latest_level - prev_level) / prev_level * 100.0
        # days of water available = storage_m3 / outflow_m3_per_day
        outflow = latest["outflow_m3_per_day"] if not pd.isna(latest["outflow_m3_per_day"]) else None
        storage = latest["storage_m3"] if not pd.isna(latest["storage_m3"]) else None
        days_avail = None
        if storage is not None and outflow is not None and outflow > 0:
            days_avail = storage / outflow
        # forecast summary
        fc = forecasts_map.get(r)
        avg_fc = round(float(fc["yhat"].mean()),2) if (fc is not None and not fc["yhat"].isna().all()) else None
        slope = float(fc["yhat"].iloc[-1] - fc["yhat"].iloc[0]) if (fc is not None and len(fc) >= 2) else 0
        if slope > 0.5:
            trend = "Rising"
        elif slope < -0.5:
            trend = "Falling"
        else:
            trend = "Stable"
        rows.append({
            "reservoir_name": r,
            "latest_date": latest["date"].strftime("%d-%m-%Y") if isinstance(latest["date"], (datetime,)) else (latest["date"].strftime("%d-%m-%Y") if isinstance(latest["date"], (pd.Timestamp,)) else str(latest["date"])),
            "latest_level_ft": latest_level,
            "previous_day_level": prev_level,
            "change_percent": round(change_percent,4) if change_percent is not None else None,
            "percent_full": latest["percent_full"],
            "risk_category": latest["risk_category"],
            "warning_low": bool(latest["warning_low"]),
            "days_of_water_available": round(days_avail,2) if days_avail is not None else None,
            "avg_forecast_7d": avg_fc,
            "forecast_direction_7d": trend
        })
    return pd.DataFrame(rows)

# ---------------- MAIN ----------------
def run_pipeline():
    print("Pipeline start:", datetime.now())

    # 1) Load master.xlsx (UiPath appends daily rows to this file)
    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError(f"Master file not found: {MASTER_FILE}")
    master_raw = pd.read_excel(MASTER_FILE)
    print("Loaded master rows:", len(master_raw))

    # 2) Normalize master (master already contains historical rows)
    master_norm = normalize_master_df(master_raw)

    # 3) Build cleaned master (all rows, compute missing values & deltas)
    cleaned_master = build_cleaned_master(master_norm)

    # Format date as DD-MM-YYYY (string) for final cleaned_master output
    cleaned_out = cleaned_master.copy()
    cleaned_out["date"] = pd.to_datetime(cleaned_out["date"], errors="coerce").dt.strftime("%d-%m-%Y")

    # Save cleaned master
    try:
        safe_save_csv(cleaned_out, CLEANED_MASTER_CSV)
        cleaned_master.to_excel(CLEANED_MASTER_XLSX, index=False)
        print("Saved cleaned master:", CLEANED_MASTER_CSV)
    except Exception as e:
        print("Warning: could not save cleaned master:", e)

    # 4) Forecasting: build series from master_norm (use proper datetime index)
    # master_norm date column is date; use datetime index
    master_norm_dt = master_norm.copy()
    master_norm_dt["date"] = pd.to_datetime(master_norm_dt["date"], errors="coerce")
    master_norm_dt = master_norm_dt.sort_values(["reservoir_name","date"]).drop_duplicates(subset=["reservoir_name","date"], keep="last").reset_index(drop=True)

    reservoirs = sorted(cleaned_master["reservoir_name"].dropna().unique().tolist())
    forecasts_map = {}
    all_forecasts = []

    for r in reservoirs:
        series_df = master_norm_dt[master_norm_dt["reservoir_name"]==r].copy()
        if series_df.empty:
            continue
        series = series_df.set_index(pd.to_datetime(series_df["date"]))["current_level_ft"].dropna()
        if series.empty:
            continue
        fc = sarimax_forecast(series, periods=FORECAST_DAYS)
        if fc is None:
            continue
        fc_out = fc.rename(columns={"ds":"date","yhat":"yhat","yhat_lower":"yhat_lower","yhat_upper":"yhat_upper"})
        fc_out["reservoir_name"] = r
        per_forecast_path = os.path.join(FORECAST_DIR, f"{sanitize_filename(r)}_forecast.csv")
        safe_save_csv(fc_out[["date","reservoir_name","yhat","yhat_lower","yhat_upper"]], per_forecast_path)
        forecasts_map[r] = fc_out
        all_forecasts.append(fc_out[["date","reservoir_name","yhat","yhat_lower","yhat_upper"]])
        print("Saved forecast for:", r)

    if all_forecasts:
        combined_fc = pd.concat(all_forecasts, ignore_index=True, sort=False)
        # format forecast date as dd-mm-yyyy
        combined_fc["date"] = pd.to_datetime(combined_fc["date"]).dt.strftime("%d-%m-%Y")
        safe_save_csv(combined_fc, FORECAST_COMBINED)
        print("Saved combined forecasts:", FORECAST_COMBINED)
    else:
        print("No forecasts generated.")

    # 5) Build insights (latest day) with previous day, change%, days available
    insights_df = build_insights(cleaned_master, forecasts_map)
    if not insights_df.empty:
        safe_save_csv(insights_df, INSIGHT_COMBINED)
        # also per-reservoir insight files
        for _, row in insights_df.iterrows():
            p = os.path.join(INSIGHT_DIR, f"{sanitize_filename(row['reservoir_name'])}_insights.csv")
            pd.DataFrame([row]).to_csv(p, index=False)
        print("Saved combined insights:", INSIGHT_COMBINED)
    else:
        print("No insights generated.")

    # 6) Summary
    summary = {
        "run_time": str(datetime.now()),
        "n_reservoirs": len(reservoirs),
        "forecasts_generated": len(all_forecasts),
        "insights_generated": len(insights_df) if not insights_df.empty else 0,
        "forecast_days": FORECAST_DAYS
    }
    try:
        with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print("Saved summary:", SUMMARY_JSON)
    except Exception:
        pass

    print("Pipeline finished. Summary:", summary)

if __name__ == "__main__":
    run_pipeline()
