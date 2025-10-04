# event_study.py
# Usage:
#   python event_study.py --event_date 2021-03-23 --pre_days 120 --post_days 20 --car_k1 5 --car_k2 10

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paths ----
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = (os.path.join(os.path.dirname(ROOT), "data", "raw")
            if os.path.basename(ROOT) == "src"
            else os.path.join(ROOT, "data", "raw"))
RES_TAB = (os.path.join(os.path.dirname(ROOT), "results", "tables")
           if os.path.basename(ROOT) == "src"
           else os.path.join(ROOT, "results", "tables"))
RES_FIG = (os.path.join(os.path.dirname(ROOT), "results", "figures")
           if os.path.basename(ROOT) == "src"
           else os.path.join(ROOT, "results", "figures"))
os.makedirs(RES_TAB, exist_ok=True)
os.makedirs(RES_FIG, exist_ok=True)

MERGED = os.path.join(DATA_DIR, "merged_market_daily.csv")

MARKET_TICKER = "^NSEI"
MACROS = {"BZ=F","INR=X","^INDIAVIX","^VIX", MARKET_TICKER}

def logret(s: pd.Series) -> pd.Series:
    return np.log(s).diff()

def fit_market_model(ret_i: pd.Series, ret_m: pd.Series):
    X = np.column_stack([np.ones(len(ret_m)), ret_m.values])
    y = ret_i.values
    alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return alpha, beta

def main(event_date: str, pre_days: int, post_days: int, car_k1: int, car_k2: int):
    if not os.path.exists(MERGED):
        raise SystemExit(f"Missing {MERGED}. Run data_download.py first.")
    df = pd.read_csv(MERGED, parse_dates=["Date"], dayfirst=False)
    df = df.rename(columns={"Date":"date"})

    # normalize date column name
    if "Date" in df.columns: df = df.rename(columns={"Date":"date"})
    df = df.set_index("date").sort_index()

    # infer equity tickers = all columns minus macros/market
    cols = list(df.columns)
    if MARKET_TICKER not in cols:
        raise SystemExit(f"Market ticker {MARKET_TICKER} not found. Columns sample: {cols[:10]}")
    tickers = [c for c in cols if c not in MACROS]

    # compute returns
    ret = df[[MARKET_TICKER] + tickers].apply(logret).dropna()

    # align event date to trading day
    ev = pd.to_datetime(event_date)
    if ev not in ret.index:
        later = ret.index[ret.index >= ev]
        ev = later[0] if len(later) else ret.index[-1]

    all_dates = ret.index
    ev_idx = all_dates.get_loc(ev)

    # windows: estimation on [-pre, -21]; event plotted from [-5, +post]
    est_start = max(0, ev_idx - (pre_days + 21))
    est_end   = max(0, ev_idx - 21)
    event_start = max(0, ev_idx - 5)
    event_end   = min(len(all_dates)-1, ev_idx + post_days)

    est_window = all_dates[est_start:est_end]
    event_window = all_dates[event_start:event_end+1]

    ret_m = ret.loc[est_window, MARKET_TICKER]

    rows = []
    ar_panels = []

    for t in tickers:
        # fit on estimation window
        mm = pd.concat([ret.loc[est_window, t], ret_m], axis=1, join="inner").dropna()
        if mm.empty: 
            continue
        alpha, beta = fit_market_model(mm.iloc[:,0], mm.iloc[:,1])

        # AR on event window
        re = ret.loc[event_window, [t, MARKET_TICKER]].dropna()
        exp = alpha + beta * re[MARKET_TICKER]
        ar = re[t] - exp
        car = ar.cumsum()

        tmp = pd.DataFrame({"date": ar.index, "ticker": t, "ar": ar.values, "car": car.values})
        ar_panels.append(tmp)

        k1_idx = min(ev_idx + car_k1, len(all_dates)-1)
        k2_idx = min(ev_idx + car_k2, len(all_dates)-1)
        k1_date = all_dates[k1_idx]; k2_date = all_dates[k2_idx]
        k1_car = car.loc[car.index <= k1_date].iloc[-1] if not car.empty else np.nan
        k2_car = car.loc[car.index <= k2_date].iloc[-1] if not car.empty else np.nan

        rows.append({"ticker": t, "alpha": alpha, "beta": beta,
                     f"CAR_{car_k1}d": k1_car, f"CAR_{car_k2}d": k2_car})

    summary = pd.DataFrame(rows).sort_values(f"CAR_{car_k2}d", ascending=False)
    panel = pd.concat(ar_panels, ignore_index=True) if ar_panels else pd.DataFrame()

    # save tables
    sum_path = os.path.join(RES_TAB, f"event_study_summary_{event_date}.csv")
    pan_path = os.path.join(RES_TAB, f"event_study_panel_{event_date}.csv")
    summary.to_csv(sum_path, index=False)
    if not panel.empty:
        panel.to_csv(pan_path, index=False)
    print(f"[OK] Saved summary: {sum_path}")
    if not panel.empty: print(f"[OK] Saved panel:   {pan_path}")

    # figures
    if not summary.empty:
        fig1 = plt.figure(figsize=(12,6))
        x = np.arange(len(summary))
        w = 0.4
        plt.bar(x - w/2, summary[f"CAR_{car_k1}d"], width=w, label=f"CAR {car_k1}d")
        plt.bar(x + w/2, summary[f"CAR_{car_k2}d"], width=w, label=f"CAR {car_k2}d")
        plt.xticks(x, summary["ticker"], rotation=75, ha="right")
        plt.title(f"NIFTY50: CAR at {car_k1}d and {car_k2}d (Event {event_date})")
        plt.legend()
        plt.tight_layout()
        fig1_path = os.path.join(RES_FIG, f"car_bar_{event_date}.png")
        fig1.savefig(fig1_path, dpi=150); plt.close(fig1)
        print(f"[OK] Saved figure: {fig1_path}")

    if not panel.empty:
        mean_ar = (panel.groupby("date")["ar"].mean()).sort_index()
        fig2 = plt.figure(figsize=(12,4))
        plt.plot(mean_ar.index, mean_ar.values, linewidth=2)
        plt.axvline(pd.to_datetime(event_date), linestyle="--")
        plt.title("Mean Abnormal Return across NIFTY50 (event window)")
        plt.xlabel("Date"); plt.ylabel("Mean AR")
        plt.tight_layout()
        fig2_path = os.path.join(RES_FIG, f"mean_ar_{event_date}.png")
        fig2.savefig(fig2_path, dpi=150); plt.close(fig2)
        print(f"[OK] Saved figure: {fig2_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_date", type=str, default="2021-03-23")
    ap.add_argument("--pre_days", type=int, default=120)
    ap.add_argument("--post_days", type=int, default=20)
    ap.add_argument("--car_k1", type=int, default=5)
    ap.add_argument("--car_k2", type=int, default=10)
    args = ap.parse_args()
    main(args.event_date, args.pre_days, args.post_days, args.car_k1, args.car_k2)
