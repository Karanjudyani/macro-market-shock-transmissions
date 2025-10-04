# data_download.py
# Usage:
#   python src/data_download.py --start 2020-12-01 --end 2021-06-30
#
# Outputs:
#   data/raw/merged_market_daily.csv     (prices, wide)
#   data/raw/ticker_sectors.csv          (ticker, sector, industry)

import argparse, os, time
import pandas as pd
import yfinance as yf

# ---- Paths ----
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = (os.path.join(os.path.dirname(ROOT), "data", "raw")
            if os.path.basename(ROOT) == "src"
            else os.path.join(ROOT, "data", "raw"))
os.makedirs(DATA_DIR, exist_ok=True)

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
                print(f"[WARN] no data for {t}")
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
            # yfinance sometimes needs a small pause to avoid rate limits
            info = yf.Ticker(t).get_info()  # newer yfinance
            sector = info.get("sector") or info.get("industryDisp") or info.get("industry")
            industry = info.get("industry") or info.get("industryDisp")
            time.sleep(0.15)
        except Exception as e:
            pass
        if not sector:
            sector = FALLBACK_SECTOR.get(t, "Unmapped")  # last resort
        rows.append({"ticker": t, "sector": sector, "industry": industry if industry else ""})
    meta = pd.DataFrame(rows)
    return meta

def main(start, end):
    # 1) Market + macros
    base = fetch_yahoo_prices([INDEX_TICKER] + MACRO_TICKERS, start, end)

    # 2) Equities
    eq = fetch_yahoo_prices(NIFTY50, start, end)

    # 3) Merge
    merged = base.join(eq, how="outer").ffill()
    merged_path = os.path.join(DATA_DIR, "merged_market_daily.csv")
    merged.to_csv(merged_path, index=True)
    print(f"[OK] Saved merged dataset {merged.shape} → {merged_path}")

    # 4) Sector metadata
    meta = fetch_sectors([c for c in eq.columns])  # only those successfully fetched
    meta_path = os.path.join(DATA_DIR, "ticker_sectors.csv")
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
