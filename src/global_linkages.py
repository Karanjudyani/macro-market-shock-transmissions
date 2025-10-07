# src/global_linkages.py
# Quantifies global linkages: how Brent / VIX / INR shocks around the event
# load into post-minus-pre changes in firm-level volatility (Δσ).
#
# Usage:
#   python src/global_linkages.py --event_date 2021-03-23
#   python src/global_linkages.py --event_date 2021-04-03 --k_pre 7 --k_post 7

import argparse, os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------- Paths --------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(ROOT) if os.path.basename(ROOT) == "src" else ROOT
DATA = os.path.join(BASE, "data", "raw")
RES_TAB = os.path.join(BASE, "results", "tables")
RES_FIG = os.path.join(BASE, "results", "figures")
os.makedirs(RES_TAB, exist_ok=True)
os.makedirs(RES_FIG, exist_ok=True)

# ---------------- Schema helpers ----------------
def normalize_vol_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either {pre_mean_sigma, post_mean_sigma, delta_sigma}
    or {sigma_pre, sigma_post, d_sigma}. Return standardized names."""
    cols = set(df.columns.str.lower())
    rename_map = {}
    if "d_sigma" in cols or "delta_sigma" in cols:
        if "delta_sigma" in cols and "d_sigma" not in cols:
            rename_map["delta_sigma"] = "d_sigma"
    if ("sigma_pre" in cols and "sigma_post" in cols) or \
       ("pre_mean_sigma" in cols and "post_mean_sigma" in cols):
        # map pre/post if needed
        if "pre_mean_sigma" in cols and "sigma_pre" not in cols:
            rename_map["pre_mean_sigma"] = "sigma_pre"
        if "post_mean_sigma" in cols and "sigma_post" not in cols:
            rename_map["post_mean_sigma"] = "sigma_post"

    if rename_map:
        df = df.rename(columns={k: v for k, v in rename_map.items()})

    # If d_sigma missing but sigma_pre/sigma_post exist, compute it
    if "d_sigma" not in df.columns and {"sigma_pre", "sigma_post"}.issubset(df.columns):
        df["d_sigma"] = df["sigma_post"].astype(float) - df["sigma_pre"].astype(float)

    need = {"ticker", "sector", "d_sigma"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERR] volatility_summary missing columns: {missing}. "
                         f"Found: {df.columns.tolist()}")
    return df[["ticker", "sector", "d_sigma"]].copy()

# ---------------- Exposure flags ----------------
def build_exposures(sector: str):
    s = str(sector or "").lower()
    energy_exp = int(any(k in s for k in ["energy", "oil", "gas"]))
    trade_exp  = int(any(k in s for k in [
        "industrial", "basic", "materials", "metal", "mining",
        "consumer cyclical", "transport", "infra", "port", "shipping"]))
    defensive  = int(any(k in s for k in [
        "consumer defensive", "fmcg", "healthcare", "pharma", "utilities", "telecom"]))
    # cyclical proxy: anything not defensive but exposed to energy/trade/auto/financials
    treated = int((not defensive) and (
        energy_exp or trade_exp or
        any(k in s for k in ["auto", "automobile", "bank", "financial services"])
    ))
    risk_exp = int(any(k in s for k in ["financial", "bank", "nbfc", "broker"]))
    fx_exp   = int(any(k in s for k in ["it", "software", "technology", "tech"]))
    return treated, energy_exp, risk_exp, fx_exp

# ---------------- Macro shock builder ----------------
def compute_macro_shocks(merged: pd.DataFrame, event_dt: pd.Timestamp, k_pre=5, k_post=5):
    """
    ΔBrent, ΔVIX, ΔINR computed as the % change from the average over the last k_pre
    trading days BEFORE event_dt to the average over the first k_post trading days
    ON/AFTER event_dt.
    """
    for col in ["BZ=F", "^VIX", "INR=X"]:
        if col not in merged.columns:
            raise SystemExit(f"[ERR] merged_market_daily.csv missing column: {col}")

    merged = merged.sort_index().copy()

    pre = merged.loc[merged.index < event_dt].tail(k_pre)
    post = merged.loc[merged.index >= event_dt].head(k_post)
    if pre.empty or post.empty:
        raise SystemExit("[ERR] Not enough days to compute macro shocks. "
                         f"Check your merged data around {event_dt.date()}.")

    out = {}
    for sym, name in [("BZ=F", "brent"), ("^VIX", "vix"), ("INR=X", "inr")]:
        pre_m, post_m = pre[sym].mean(), post[sym].mean()
        out[name] = (post_m - pre_m) / pre_m
    return pd.Series(out)

# ----------------------------- Main -----------------------------
def main(event_date: str, k_pre: int, k_post: int):
    ev = pd.to_datetime(event_date)

    # 1) Load volatility summary and standardize schema
    vol_path = os.path.join(RES_TAB, f"volatility_summary_{event_date}.csv")
    if not os.path.exists(vol_path):
        raise SystemExit(f"[ERR] Not found: {vol_path}")
    vol_raw = pd.read_csv(vol_path)
    vol = normalize_vol_summary(vol_raw)

    # 2) Ensure sectors present (merge metadata if needed)
    meta_path = os.path.join(DATA, "ticker_sectors.csv")
    if os.path.exists(meta_path) and "sector" in vol.columns:
        # Only fill missing sectors from meta
        meta = pd.read_csv(meta_path)
        if {"ticker", "sector"}.issubset(meta.columns):
            fill = meta[["ticker", "sector"]].drop_duplicates("ticker")
            vol = vol.merge(fill, on="ticker", how="left", suffixes=("", "_meta"))
            vol["sector"] = vol["sector"].fillna(vol["sector_meta"])
            vol = vol.drop(columns=[c for c in ["sector_meta"] if c in vol.columns])

    # 3) Load merged prices to compute macro shocks
    merged_path = os.path.join(DATA, "merged_market_daily.csv")
    merged = pd.read_csv(merged_path, parse_dates=["Date"]).set_index("Date")
    dmac = compute_macro_shocks(merged, ev, k_pre=k_pre, k_post=k_post)
    d_brent, d_vix, d_inr = dmac["brent"], dmac["vix"], dmac["inr"]
    print(f"[INFO] Macro shocks ({k_pre}d pre → {k_post}d post averages): "
          f"ΔBrent={d_brent:.2%}, ΔVIX={d_vix:.2%}, ΔINR={d_inr:.2%}")

    # 4) Build exposures
    expo = vol["sector"].apply(build_exposures)
    vol[["Treated", "EnergyExp", "RiskExp", "FXExp"]] = pd.DataFrame(expo.tolist(), index=vol.index)

    # 5) Design matrix + OLS
    # Cross-section: d_sigma_i = a + b1*Treated_i + b2*(EnergyExp_i * ΔBrent) +
    #                              b3*(RiskExp_i * ΔVIX) + b4*(FXExp_i * ΔINR) + ε_i
    X = pd.DataFrame({
        "const": 1.0,
        "Treated": vol["Treated"].astype(int),
        "EnergyExp_x_dBrent": vol["EnergyExp"].astype(int) * d_brent,
        "RiskExp_x_dVIX":     vol["RiskExp"].astype(int)   * d_vix,
        "FXExp_x_dINR":       vol["FXExp"].astype(int)     * d_inr,
    })
    y = vol["d_sigma"].astype(float)

    model = sm.OLS(y, X).fit(cov_type="HC1")
    print(model.summary())

    # 6) Save coefficient table
    out_tbl = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p": model.pvalues
    })
    coef_path = os.path.join(RES_TAB, f"global_linkages_{event_date}.csv")
    out_tbl.to_csv(coef_path)
    print(f"[OK] Saved coefficients → {coef_path}")

    # 7) Quick visuals: exposure vs Δσ (with fitted two-point line)
    def two_point_plot(xflag, slope_key, title, fname):
        plt.figure(figsize=(6,4))
        plt.scatter(vol[xflag]*1.0, y, alpha=0.6)
        xline = np.array([0,1], dtype=float)
        # Prediction line across exposure 0→1:
        # E[Δσ | exposure=x] = const + slope*x
        yline = model.params["const"] + model.params[slope_key]*xline
        plt.plot(xline, yline, linewidth=2)
        plt.xticks([0,1], ["No exposure","Exposure"])
        plt.ylabel("Δ volatility (post − pre)")
        plt.title(title)
        plt.tight_layout()
        out = os.path.join(RES_FIG, fname)
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[OK] Saved figure → {out}")

    two_point_plot("EnergyExp", "EnergyExp_x_dBrent",
                   f"Energy exposure vs Δσ  (ΔBrent={d_brent:.1%})\nEvent: {event_date}",
                   f"global_energy_scatter_{event_date}.png")

    two_point_plot("FXExp", "FXExp_x_dINR",
                   f"FX exposure vs Δσ  (ΔINR={d_inr:.1%})\nEvent: {event_date}",
                   f"global_fx_scatter_{event_date}.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_date", type=str, required=True, help="YYYY-MM-DD (e.g., 2021-03-23)")
    ap.add_argument("--k_pre", type=int, default=5, help="business days averaged before the event")
    ap.add_argument("--k_post", type=int, default=5, help="business days averaged on/after the event")
    args = ap.parse_args()
    main(args.event_date, args.k_pre, args.k_post)
