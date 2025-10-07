# data_download.py
# Usage:
#   python src/data_download.py --start 2020-12-01 --end 2021-06-30
#
# Outputs:
#   data/raw/merged_market_daily.csv     (prices, wide)
#   data/raw/ticker_sectors.csv          (ticker, sector, industry, exposure_group, source)
#   data/raw/all_tickers.csv             (union of NIFTY50 + extra list)

import argparse, os, time
import pandas as pd
import yfinance as yf
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parent
DATA_DIR = (ROOT.parent / "data" / "raw") if ROOT.name == "src" else (ROOT / "data" / "raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Universe: NIFTY 50 tickers ----
NIFTY50 = [
    "ADANIPORTS.NS","ASIANPAINT.NS","AXISBANK.NS","BAJAJ-AUTO.NS","BAJFINANCE.NS",
    "BAJAJFINSV.NS","BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS","GRASIM.NS",
    "HCLTECH.NS","HDFCBANK.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","INDUSINDBK.NS","INFY.NS","IOC.NS","ITC.NS",
    "JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","M&M.NS","MARUTI.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS","RELIANCE.NS",
    "SBIN.NS","SHREECEM.NS","SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS",
    "TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS",
    "UPL.NS","WIPRO.NS"
]

INDEX_TICKER = "^NSEI"   # Market
MACRO_TICKERS = ["BZ=F", "INR=X", "^INDIAVIX", "^VIX"]  # Brent, USD/INR, India VIX, CBOE VIX

# ---- Fallback sector map (used if yfinance doesn't return sector) ----
FALLBACK_SECTOR = {
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
    "HINDUNILVR.NS":"FMCG","ITC.NS":"FMCG","NESTLEIND.NS":"FMCG","TATACONSUM.NS":"FMCG","BRITANNIA.NS":"FMCG",
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
    # Financials (Non-Bank)
    "BAJFINANCE.NS":"Finance","BAJAJFINSV.NS":"Finance",
}

def fetch_yahoo_prices(tickers, start, end):
    out = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                print(f"[WARN] No data for {t}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            out[t] = df["Close"].rename(t)
        except Exception as e:
            print(f"[WARN] {t} failed: {e}")
    if not out:
        raise SystemExit("No data fetched. Check tickers or internet.")
    return pd.DataFrame(out).sort_index()

def fetch_sectors(tickers):
    rows = []
    for t in tickers:
        sector, industry = None, None
        try:
            info = yf.Ticker(t).get_info()  # newer yfinance
            sector = info.get("sector") or info.get("industryDisp") or info.get("industry")
            industry = info.get("industry") or info.get("industryDisp")
            time.sleep(0.15)  # be gentle to the API
        except Exception:
            pass
        if not sector:
            sector = FALLBACK_SECTOR.get(t, "Unmapped")  # last resort
        rows.append({"ticker": t, "sector": sector, "industry": industry if industry else ""})
    return pd.DataFrame(rows)

def main(start, end):
    # --- Load extra exposed universe if present ---
    extra_path = DATA_DIR / "extra_exposed_tickers.csv"
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        if "ticker" not in extra.columns:
            raise SystemExit("extra_exposed_tickers.csv must have a 'ticker' column.")
        extra["source"] = "extra"
        # normalize col
        if "exposure_group" not in extra.columns:
            extra["exposure_group"] = "unknown"
    else:
        extra = pd.DataFrame(columns=["ticker","exposure_group","source"])

    # Base NIFTY list → DataFrame
    base = pd.DataFrame({"ticker": NIFTY50, "source": "nifty50"})
    base["exposure_group"] = "none"

    # Union
    universe = pd.concat([base, extra], ignore_index=True).drop_duplicates(subset=["ticker"])
    universe.to_csv(DATA_DIR / "all_tickers.csv", index=False)
    print(f"[INFO] Loaded {len(universe)} tickers (by source: {universe['source'].value_counts().to_dict()})")

    # 1) Market + macros
    base_df = fetch_yahoo_prices([INDEX_TICKER] + MACRO_TICKERS, start, end)

    # 2) Equities
    eq_df = fetch_yahoo_prices(universe["ticker"].tolist(), start, end)

    # 3) Merge (market+macro, then equities)
    merged = base_df.join(eq_df, how="outer").ffill()
    merged_path = DATA_DIR / "merged_market_daily.csv"
    merged.to_csv(merged_path, index=True)
    print(f"[OK] Saved merged dataset {merged.shape} → {merged_path}")

    # 4) Sector metadata (only for successfully fetched equities)
    fetched_equities = [c for c in eq_df.columns]
    meta = fetch_sectors(fetched_equities)

    # Add exposure_group and source info from universe
    meta = meta.merge(universe[["ticker","exposure_group","source"]], on="ticker", how="left")
    meta["exposure_group"] = meta["exposure_group"].fillna("none")
    meta["source"] = meta["source"].fillna("unknown")

    meta_path = DATA_DIR / "ticker_sectors.csv"
    meta.to_csv(meta_path, index=False)

    # Warn if any are unmapped
    unmapped = meta[meta["sector"] == "Unmapped"]["ticker"].tolist()
    if unmapped:
        print(f"[WARN] Unmapped tickers (set via FALLBACK_SECTOR to fix): {unmapped}")
    print(f"[OK] Saved ticker→sector map ({len(meta)} rows) → {meta_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2020-12-01")
    ap.add_argument("--end", type=str, default="2021-06-30")
    args = ap.parse_args()
    main(args.start, args.end)

