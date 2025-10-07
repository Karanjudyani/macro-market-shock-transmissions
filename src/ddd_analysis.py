# src/ddd_analysis.py
# Triple-Difference estimation with firm and date FE.
# Requires:
#   results/tables/event_study_panel_YYYY-MM-DD.csv  (has date,ticker,ar,car)
#   data/raw/ticker_sectors.csv                      (ticker, sector, industry, [exposure_group optional])

import argparse, os
import pandas as pd
import statsmodels.formula.api as smf

# --------------------- config ---------------------
# Sector buckets for Treated vs Defensive
TREATED_SECTORS = {"Metals", "Energy", "Autos", "Infrastructure", "Chemicals"}
DEFENSIVE_SECTORS = {"FMCG", "Pharma", "Utilities", "Telecom", "IT", "Cement", "Finance", "Banks"}

# If your ticker_sectors.csv does NOT have exposure_group, we will create it via this set:
HIGH_EXPOSURE_TICKERS = {
    "SCI.NS", "PETRONET.NS", "BPCL.NS", "ONGC.NS", "IOC.NS",
    "CHAMBLFERT.NS", "GODREJAGRO.NS", "COROMANDEL.NS"
}

# --------------------- helpers ---------------------
def pick(cols, candidates, required=True):
    """Return the first column name in `candidates` that appears in `cols`."""
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"Missing expected column from {candidates}. Have: {list(cols)}")
    return None

def classify_group(sector):
    if pd.isna(sector): return "Other"
    if sector in TREATED_SECTORS: return "Treated"
    if sector in DEFENSIVE_SECTORS: return "Defensive"
    return "Other"

# --------------------- paths ---------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
TABLES = os.path.join(PROJECT, "results", "tables")
DATA   = os.path.join(PROJECT, "data", "raw")
os.makedirs(TABLES, exist_ok=True)

# --------------------- main ---------------------
def main(event_date, drop_start=None, drop_end=None):
    # 1) Load panel (event-time returns)
    panel_path = os.path.join(TABLES, f"event_study_panel_{event_date}.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"{panel_path} not found. Run event_study.py for {event_date} first.")
    df = pd.read_csv(panel_path)
    print(f"[INFO] Loaded panel {df.shape}, columns={list(df.columns)}")

    # Standardize expected column names (lower/upper safe)
    date_col   = pick(df.columns, ["date", "Date"])
    ticker_col = pick(df.columns, ["ticker", "Ticker", "symbol", "Symbol"])
    ar_col     = pick(df.columns, ["AR", "ar", "abnormal_return"], required=True)
    df[date_col] = pd.to_datetime(df[date_col])

    # 2) Load sector/exposure metadata
    meta_path = os.path.join(DATA, "ticker_sectors.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found. Make sure data_download.py created it.")
    meta = pd.read_csv(meta_path)
    # Ensure required columns exist
    if "ticker" not in meta.columns:
        raise KeyError("ticker_sectors.csv must contain a 'ticker' column.")
    if "sector" not in meta.columns:
        raise KeyError("ticker_sectors.csv must contain a 'sector' column.")

    # If exposure_group missing, create it from HIGH_EXPOSURE_TICKERS set
    if "exposure_group" not in meta.columns:
        meta["exposure_group"] = meta["ticker"].apply(
            lambda t: "high" if t in HIGH_EXPOSURE_TICKERS else "none"
        )

    # 3) Merge meta into panel
    df = df.merge(meta[["ticker", "sector", "exposure_group"]],
                  on="ticker", how="left")

    # 4) Build flags
    df["Group"] = df["sector"].apply(classify_group)
    df = df[df["Group"].isin(["Treated", "Defensive"])].copy()

    df["TreatedFlag"]  = (df["Group"] == "Treated").astype(int)
    df["HighExposure"] = df["exposure_group"].isin(["high", "shipping", "oil", "agri"]).astype(int)

    EVENT_DATE = pd.to_datetime(event_date)
    df["Post"] = (df[date_col] >= EVENT_DATE).astype(int)

    # Keep only rows with AR
    df = df.dropna(subset=[ar_col]).copy()
    df["y"] = df[ar_col]

    # Optional donut exclusion (e.g., to trim COVID news overlap)
    if drop_start and drop_end:
        drop_start = pd.to_datetime(drop_start)
        drop_end   = pd.to_datetime(drop_end)
        m = (df[date_col] >= drop_start) & (df[date_col] <= drop_end)
        n_drop = int(m.sum())
        df = df.loc[~m].copy()
        print(f"[INFO] Dropped donut window {drop_start.date()}–{drop_end.date()}: {n_drop} rows")

    # Narrow to a reasonable event window (if your panel is wider)
    # (comment out if your panel is already limited)
    rel_days = (df[date_col] - EVENT_DATE).dt.days
    df = df[(rel_days >= -10) & (rel_days <= 40)].copy()

    # 5) Standardize names for the regression (patsy-friendly)
    df = df.rename(columns={ticker_col: "tic", date_col: "dt"})

    # 6) Estimate DDD with firm FE and date FE; cluster SEs by ticker
    formula = "y ~ TreatedFlag*HighExposure*Post + C(tic) + C(dt)"
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["tic"]}
    )

    # 7) Save output table
    out_csv = os.path.join(TABLES, f"ddd_summary_{event_date}.csv")
    coefs = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "pval": model.pvalues.values,
    })
    coefs.to_csv(out_csv, index=False)
    print(f"[OK] Saved DDD summary → {out_csv}\n")
    print(model.summary().tables[1])
    key = "TreatedFlag:HighExposure:Post"
    if key in model.params.index:
        print(f"\n[KEY] {key} = {model.params[key]:.6f}  (p={model.pvalues[key]:.3f})")
        print("Interpretation: extra daily abnormal return for treated & high-exposure firms AFTER the event,\n"
              "net of stock and date fixed effects. Negative & significant ⇒ Suez transmission.\n")
    else:
        print(f"\n[WARN] {key} not found in model terms (check column flags).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_date", required=True, type=str)     # e.g., 2021-03-23
    parser.add_argument("--drop_start", default=None, type=str)
    parser.add_argument("--drop_end", default=None, type=str)
    args = parser.parse_args()
    main(args.event_date, args.drop_start, args.drop_end)
