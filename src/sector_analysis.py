# 03_sector_analysis.py
# Usage:
#   python src/03_sector_analysis.py --event_date 2021-03-23
#
# Reads:
#   results/tables/event_study_summary_<EVENT>.csv
#
# Writes:
#   results/tables/sector_avg_mean_<EVENT>.csv
#   results/tables/sector_avg_median_<EVENT>.csv
#   results/figures/sector_bar_mean_<EVENT>.png
#   results/figures/sector_bar_median_<EVENT>.png
#   results/figures/sector_box_car10_<EVENT>.png
#   results/tables/treated_vs_defensive_<EVENT>.txt  (with simple Welch t-stat)

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- paths ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
RES_TAB = (os.path.join(os.path.dirname(ROOT), "results", "tables")
           if os.path.basename(ROOT) == "src" else os.path.join(ROOT, "results", "tables"))
RES_FIG = (os.path.join(os.path.dirname(ROOT), "results", "figures")
           if os.path.basename(ROOT) == "src" else os.path.join(ROOT, "results", "figures"))
os.makedirs(RES_TAB, exist_ok=True)
os.makedirs(RES_FIG, exist_ok=True)

# ---------- curated sector mapping (editted) ----------
SECTOR_MAP = {
    # Banks
    "HDFCBANK.NS":"Banks","ICICIBANK.NS":"Banks","AXISBANK.NS":"Banks","KOTAKBANK.NS":"Banks","SBIN.NS":"Banks","INDUSINDBK.NS":"Banks",
    # IT
    "TCS.NS":"IT","INFY.NS":"IT","HCLTECH.NS":"IT","TECHM.NS":"IT","WIPRO.NS":"IT",
    # Pharma
    "SUNPHARMA.NS":"Pharma","CIPLA.NS":"Pharma","DRREDDY.NS":"Pharma","DIVISLAB.NS":"Pharma",
    # Metals & Mining
    "TATASTEEL.NS":"Metals","JSWSTEEL.NS":"Metals","HINDALCO.NS":"Metals","COALINDIA.NS":"Metals",
    # Energy / O&G
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy","BPCL.NS":"Energy","IOC.NS":"Energy",
    # Autos
    "MARUTI.NS":"Autos","TATAMOTORS.NS":"Autos","HEROMOTOCO.NS":"Autos","BAJAJ-AUTO.NS":"Autos","M&M.NS":"Autos","EICHERMOT.NS":"Autos",
    # FMCG
    "HINDUNILVR.NS":"FMCG","ASIANPAINT.NS": "FMCG","ITC.NS":"FMCG","NESTLEIND.NS":"FMCG","TATACONSUM.NS":"FMCG","BRITANNIA.NS":"FMCG",
    # Cement / Materials
    "ULTRACEMCO.NS":"Cement","SHREECEM.NS":"Cement","GRASIM.NS":"Cement",
    # Utilities / Power
    "NTPC.NS":"Utilities","POWERGRID.NS":"Utilities",
    # Telecom
    "BHARTIARTL.NS":"Telecom",
    # Infrastructure / Ports
    "ADANIPORTS.NS":"Infrastructure",
    # Consumer Discretionary / Durables
    "TITAN.NS":"Consumer Discretionary",
    # Chemicals / Agri
    "UPL.NS":"Chemicals",
    # Industrials / Capital Goods
    "LT.NS":"Industrials",
    # Finance (Non-bank)
    "BAJFINANCE.NS":"Finance","BAJAJFINSV.NS":"Finance",
}

# group buckets for a simple treated vs defensive split
TREATED_SECTORS = {"Metals","Energy","Autos","Infrastructure","Chemicals"}
DEFENSIVE_SECTORS = {"FMCG","Pharma","Utilities","Telecom","IT","Cement"}

def load_summary(event_date: str) -> pd.DataFrame:
    path = os.path.join(RES_TAB, f"event_study_summary_{event_date}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run event_study.py first.")
    df = pd.read_csv(path)
    need = {"ticker","CAR_5d","CAR_10d"}
    if not need.issubset(df.columns):
        raise SystemExit(f"Summary missing columns: {need - set(df.columns)}")
    return df

def ensure_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    if df["sector"].isna().any():
        missing = df[df["sector"].isna()]["ticker"].unique().tolist()
        raise SystemExit(f"Unmapped tickers found: {missing}\nAdd them to SECTOR_MAP and rerun.")
    return df

def sector_tables(df: pd.DataFrame):
    mean_tbl = (df.groupby("sector", as_index=False)[["CAR_5d","CAR_10d"]]
                  .mean()
                  .sort_values("CAR_10d", ascending=False))
    median_tbl = (df.groupby("sector", as_index=False)[["CAR_5d","CAR_10d"]]
                    .median()
                    .sort_values("CAR_10d", ascending=False))
    return mean_tbl, median_tbl

def plot_sector_bars(tbl: pd.DataFrame, title: str, outpath: str):
    x = np.arange(len(tbl))
    plt.figure(figsize=(12,6))
    plt.bar(x - 0.2, tbl["CAR_5d"], width=0.4, label="CAR 5d")
    plt.bar(x + 0.2, tbl["CAR_10d"], width=0.4, label="CAR 10d")
    plt.xticks(x, tbl["sector"], rotation=45, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_box_car10(df: pd.DataFrame, outpath: str):
    # boxplot of CAR_10d by sector (ordered by median)
    med_order = (df.groupby("sector")["CAR_10d"].median()
                   .sort_values(ascending=False).index.tolist())
    data = [df.loc[df["sector"]==s, "CAR_10d"].values for s in med_order]
    plt.figure(figsize=(12,6))
    plt.boxplot(data, labels=med_order, vert=True)
    plt.xticks(rotation=45, ha="right")
    plt.title("CAR 10d distribution by sector")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def welch_t(a: np.ndarray, b: np.ndarray):
    # simple Welch t-stat (unequal variances)
    a = np.asarray(a); b = np.asarray(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    tnum = ma - mb
    tden = np.sqrt(va/na + vb/nb)
    tstat = tnum / tden if tden > 0 else np.nan
    return tstat, ma, mb, na, nb

def treated_vs_defensive(df: pd.DataFrame, event_date: str):
    treated = df[df["sector"].isin(TREATED_SECTORS)]["CAR_10d"].dropna().values
    defensive = df[df["sector"].isin(DEFENSIVE_SECTORS)]["CAR_10d"].dropna().values
    tstat, mt, md, nt, nd = welch_t(treated, defensive)
    text = [
        f"Event: {event_date}",
        f"TREATED sectors: {sorted(TREATED_SECTORS)}",
        f"DEFENSIVE sectors: {sorted(DEFENSIVE_SECTORS)}",
        f"Mean CAR_10d (treated)   = {mt:.4f}  (n={nt})",
        f"Mean CAR_10d (defensive) = {md:.4f}  (n={nd})",
        f"Welch t-stat (treated - defensive) = {tstat:.3f}",
        "Note: This is a descriptive gap; formal p-values need df calc or bootstrap."
    ]
    out = os.path.join(RES_TAB, f"treated_vs_defensive_{event_date}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    return out

def main(event_date: str):
    df = load_summary(event_date)
    df = ensure_mapping(df)

    mean_tbl, median_tbl = sector_tables(df)

    mean_out = os.path.join(RES_TAB, f"sector_avg_mean_{event_date}.csv")
    median_out = os.path.join(RES_TAB, f"sector_avg_median_{event_date}.csv")
    mean_tbl.to_csv(mean_out, index=False)
    median_tbl.to_csv(median_out, index=False)

    plot_sector_bars(mean_tbl, f"Sector-average CAR (mean) — Event {event_date}",
                     os.path.join(RES_FIG, f"sector_bar_mean_{event_date}.png"))
    plot_sector_bars(median_tbl, f"Sector-average CAR (median) — Event {event_date}",
                     os.path.join(RES_FIG, f"sector_bar_median_{event_date}.png"))

    # boxplot to show dispersion
    plot_box_car10(df, os.path.join(RES_FIG, f"sector_box_car10_{event_date}.png"))

    # treated vs defensive comparison
    tvd_path = treated_vs_defensive(df, event_date)

    print(f"[OK] Saved tables:\n  {mean_out}\n  {median_out}")
    print(f"[OK] Saved figures:\n  {os.path.join(RES_FIG, f'sector_bar_mean_{event_date}.png')}\n  {os.path.join(RES_FIG, f'sector_bar_median_{event_date}.png')}\n  {os.path.join(RES_FIG, f'sector_box_car10_{event_date}.png')}")
    print(f"[OK] Saved treated vs defensive note:\n  {tvd_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_date", type=str, default="2021-03-23")
    args = ap.parse_args()
    main(args.event_date)
