# volatility_models.py
# Usage:
#   python src/volatility_models.py --event_date 2021-03-23
#   (optional) python src/volatility_models.py --event_date 2021-04-03
#
# Inputs:
#   results/tables/event_study_panel_{DATE}.csv  (columns: date, ticker, ar, car)
#   data/raw/ticker_sectors.csv                  (columns: ticker, sector, [exposure_group optional])
#
# Outputs:
#   results/tables/volatility_summary_{DATE}.csv           (per-ticker pre/post vol + delta)
#   results/tables/volatility_sector_{DATE}.csv            (sector-level means/medians)
#   results/tables/volatility_groups_{DATE}.csv            (treated vs defensive summary + t-test)
#   results/figures/vol_hist_{DATE}.png                    (histogram of Δvol)
#   results/figures/vol_sector_bar_{DATE}.png              (sector mean Δvol)
#   results/figures/vol_sector_box_{DATE}.png              (sector boxplot Δvol)

import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ---------- paths ----------
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
TABLES_DIR = os.path.join(PROJECT, "results", "tables")
FIGS_DIR = os.path.join(PROJECT, "results", "figures")
DATA_DIR = os.path.join(PROJECT, "data", "raw")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# ---------- sector buckets ----------
TREATED_SECTORS = {"Metals","Energy","Autos","Infrastructure","Chemicals"}
DEFENSIVE_SECTORS = {"FMCG","Pharma","Utilities","Telecom","IT","Cement","Finance","Banks"}

# Optional fallback for high-exposure classification if exposure_group not present
HIGH_EXPOSURE_TICKERS = {
    "SCI.NS","PETRONET.NS","BPCL.NS","ONGC.NS","IOC.NS",
    "CHAMBLFERT.NS","GODREJAGRO.NS","COROMANDEL.NS","ADANIPORTS.NS"
}

# ---------- garch helper ----------
def estimate_garch_sigma_mean(returns):
    """
    Fit GARCH(1,1) to 'returns' (pd.Series) and return mean conditional volatility.
    Returns (mean_sigma, model_ok_flag). Falls back to simple std if arch not available.
    """
    try:
        from arch import arch_model
        # Scale to percent to help optimizer stability
        am = arch_model(returns.values * 100.0, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
        res = am.fit(disp="off")
        sigma = res.conditional_volatility / 100.0  # back to raw return scale
        return float(np.mean(sigma)), True
    except Exception:
        # Fallback: robust std dev as proxy
        return float(np.std(returns.values, ddof=1)), False

def classify_group(sector: str) -> str:
    if pd.isna(sector): return "Other"
    if sector in TREATED_SECTORS: return "Treated"
    if sector in DEFENSIVE_SECTORS: return "Defensive"
    return "Other"

def main(event_date: str):
    event_date = pd.to_datetime(event_date)

    # ---- load panel ----
    panel_path = os.path.join(TABLES_DIR, f"event_study_panel_{event_date.date()}.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"{panel_path} not found. Run event_study.py for {event_date.date()} first.")
    df = pd.read_csv(panel_path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", "date")
    tic_col  = cols.get("ticker", "ticker")
    ar_col   = "ar" if "ar" in df.columns else "AR"
    df[date_col] = pd.to_datetime(df[date_col])

    # ---- load sector metadata ----
    meta_path = os.path.join(DATA_DIR, "ticker_sectors.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found. Run data_download.py first.")
    meta = pd.read_csv(meta_path)
    if "ticker" not in meta.columns or "sector" not in meta.columns:
        raise KeyError("ticker_sectors.csv must contain 'ticker' and 'sector' columns.")

    # ensure exposure_group column
    if "exposure_group" not in meta.columns:
        meta["exposure_group"] = meta["ticker"].apply(
            lambda t: "high" if t in HIGH_EXPOSURE_TICKERS else "none"
        )

    # ---- merge metadata ----
    df = df.merge(meta[["ticker","sector","exposure_group"]], left_on=tic_col, right_on="ticker", how="left")

    # keep only rows with AR
    df = df.dropna(subset=[ar_col]).copy()

    # ---- split pre & post within available panel window ----
    # your event panel is typically [-5, +40]; that’s fine for a first-pass volatility contrast
    pre_df  = df[df[date_col] <  event_date].copy()
    post_df = df[df[date_col] >= event_date].copy()

    # ---- per-ticker garch/std estimation ----
    rows = []
    used_garch_any = False
    for tic, g in df.groupby(tic_col):
        g = g.sort_values(date_col)
        pre = g[g[date_col] < event_date]
        post = g[g[date_col] >= event_date]
        if len(pre) < 5 or len(post) < 5:
            continue  # need minimally sufficient obs in both segments

        pre_sigma, ok1  = estimate_garch_sigma_mean(pre[ar_col])
        post_sigma, ok2 = estimate_garch_sigma_mean(post[ar_col])
        used_garch_any = used_garch_any or (ok1 and ok2)

        rows.append({
            "ticker": tic,
            "sector": (pre["sector"].iloc[0] if "sector" in pre.columns else np.nan),
            "exposure_group": (pre["exposure_group"].iloc[0] if "exposure_group" in pre.columns else "none"),
            "pre_mean_sigma": pre_sigma,
            "post_mean_sigma": post_sigma,
            "delta_sigma": post_sigma - pre_sigma
        })

    vol = pd.DataFrame(rows)
    if vol.empty:
        raise SystemExit("No tickers had enough pre/post observations to fit volatility. Widen your event panel.")

    # ---- group flags ----
    vol["Group"] = vol["sector"].apply(classify_group)
    vol["TreatedFlag"]  = (vol["Group"] == "Treated").astype(int)
    vol["HighExposure"] = vol["exposure_group"].isin(["high","shipping","oil","agri"]).astype(int)

    # ---- save per-ticker table ----
    out1 = os.path.join(TABLES_DIR, f"volatility_summary_{event_date.date()}.csv")
    vol.to_csv(out1, index=False)
    print(f"[OK] Saved per-ticker volatility summary → {out1}")
    if not used_garch_any:
        print("[WARN] arch package not found or failed; used std-dev fallback. Consider: pip install arch")

    # ---- sector aggregation ----
    sect = vol.groupby("sector", dropna=False).agg(
        mean_delta=("delta_sigma","mean"),
        median_delta=("delta_sigma","median"),
        count=("delta_sigma","size")
    ).reset_index().sort_values("mean_delta", ascending=False)
    out2 = os.path.join(TABLES_DIR, f"volatility_sector_{event_date.date()}.csv")
    sect.to_csv(out2, index=False)
    print(f"[OK] Saved sector-level volatility table → {out2}")

    # ---- treated vs defensive test ----
    treated = vol.loc[vol["Group"]=="Treated", "delta_sigma"]
    defens  = vol.loc[vol["Group"]=="Defensive", "delta_sigma"]
    tstat, pval = (np.nan, np.nan)
    if len(treated) > 1 and len(defens) > 1:
        tstat, pval = stats.ttest_ind(treated, defens, equal_var=False, nan_policy="omit")

    grp = pd.DataFrame({
        "group": ["Treated","Defensive","Diff(T-D)"],
        "mean_delta": [treated.mean(), defens.mean(), treated.mean() - defens.mean()],
        "median_delta": [treated.median(), defens.median(), np.nan],
        "n": [treated.count(), defens.count(), np.nan],
        "tstat_TminusD": [np.nan, np.nan, tstat],
        "pval_TminusD": [np.nan, np.nan, pval]
    })
    out3 = os.path.join(TABLES_DIR, f"volatility_groups_{event_date.date()}.csv")
    grp.to_csv(out3, index=False)
    print(f"[OK] Saved treated vs defensive summary → {out3}")

    # ---- figures ----
    # 1) Histogram of delta vol
    plt.figure(figsize=(7,5))
    plt.hist(vol["delta_sigma"].dropna(), bins=20)
    plt.axvline(0, linestyle="--")
    plt.title(f"Change in Conditional Volatility (Post–Pre)\nEvent: {event_date.date()}")
    plt.xlabel("Δ volatility (σ_post - σ_pre)")
    plt.ylabel("Number of firms")
    fig1 = os.path.join(FIGS_DIR, f"vol_hist_{event_date.date()}.png")
    plt.tight_layout()
    plt.savefig(fig1, dpi=160)
    plt.close()
    print(f"[OK] Saved figure → {fig1}")

    # 2) Sector bar of mean delta
    plt.figure(figsize=(9,5))
    splot = sect.dropna(subset=["sector"])
    plt.bar(splot["sector"], splot["mean_delta"])
    plt.axhline(0, linestyle="--")
    plt.title(f"Mean ΔVolatility by Sector (Post–Pre)\nEvent: {event_date.date()}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Δ volatility")
    fig2 = os.path.join(FIGS_DIR, f"vol_sector_bar_{event_date.date()}.png")
    plt.tight_layout()
    plt.savefig(fig2, dpi=160)
    plt.close()
    print(f"[OK] Saved figure → {fig2}")

    # 3) Sector boxplot of delta
    ordered = sect.sort_values("median_delta", ascending=False)["sector"].tolist()
    data = [vol.loc[vol["sector"]==s, "delta_sigma"].dropna().values for s in ordered]
    plt.figure(figsize=(9,6))
    plt.boxplot(data, tick_labels=ordered, vert=True)
    plt.axhline(0, linestyle="--")
    plt.title(f"ΔVolatility distribution by Sector\nEvent: {event_date.date()}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Δ volatility")
    fig3 = os.path.join(FIGS_DIR, f"vol_sector_box_{event_date.date()}.png")
    plt.tight_layout()
    plt.savefig(fig3, dpi=160)
    plt.close()
    print(f"[OK] Saved figure → {fig3}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_date", required=True, help="YYYY-MM-DD (e.g., 2021-03-23)")
    args = ap.parse_args()
    main(args.event_date)
