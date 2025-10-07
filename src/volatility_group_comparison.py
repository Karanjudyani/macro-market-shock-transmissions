# src/volatility_group_comparison.py
# Usage:
#   python src/volatility_group_comparison.py --event_date 2021-03-23
#
# Reads results/tables/volatility_summary_<date>.csv
# Compares Δvolatility (post - pre) for Treated vs Defensive groups
# Saves: results/tables/vol_vol_group_summary_<date>.csv
#        results/figures/vol_vol_group_bars_<date>.png

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(os.path.dirname(ROOT), "results")
TAB_DIR = os.path.join(RES_DIR, "tables")
FIG_DIR = os.path.join(RES_DIR, "figures")
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# --- Your preferred buckets ---
TREATED_SECTORS   = {"Metals","Energy","Autos","Infrastructure","Chemicals","Industrials","Basic Materials"}
DEFENSIVE_SECTORS = {"FMCG","Pharma","Utilities","Telecom","IT","Cement","Consumer Defensive","Technology"}

def to_group(sector: str) -> str:
    if sector in TREATED_SECTORS:
        return "Treated"
    if sector in DEFENSIVE_SECTORS:
        return "Defensive"
    return "Other"

def bootstrap_mean(arr, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(B, len(arr)), replace=True).mean(axis=1)
    return draws.mean(), np.percentile(draws, [2.5, 97.5])

def main(event_date: str):
    in_csv = os.path.join(TAB_DIR, f"volatility_summary_{event_date}.csv")
    df = pd.read_csv(in_csv)

    # Harmonize column names
    if "delta_sigma" in df.columns:
        df["d_sigma"] = df["delta_sigma"]
    elif "d_sigma" in df.columns:
        pass
    else:
        raise KeyError("Could not find Δ-vol column: expected 'delta_sigma' or 'd_sigma'.")

    # Ensure we have a clean Group using your buckets
    if "sector" not in df.columns:
        raise KeyError("Expected 'sector' in volatility summary CSV.")
    df["Group"] = df["sector"].apply(to_group)

    # Keep only Treated vs Defensive
    df = df[df["Group"].isin(["Treated","Defensive"])].copy()
    if df.empty:
        raise SystemExit("No rows after filtering to Treated/Defensive. Check sector names/buckets.")

    # Summaries
    summary = df.groupby("Group")["d_sigma"].agg(["mean","std","count"]).reset_index()
    summary_path = os.path.join(TAB_DIR, f"vol_vol_group_summary_{event_date}.csv")
    summary.to_csv(summary_path, index=False)

    # Welch t-test
    treated = df.loc[df["Group"]=="Treated", "d_sigma"].values
    defens  = df.loc[df["Group"]=="Defensive","d_sigma"].values
    tstat, pval = stats.ttest_ind(treated, defens, equal_var=False, nan_policy="omit")

    # Bootstrap CIs
    t_mean, (t_lo, t_hi) = bootstrap_mean(treated)
    d_mean, (d_lo, d_hi) = bootstrap_mean(defens)

    # Bar + CI figure
    fig, ax = plt.subplots(figsize=(7,5))
    x = np.arange(2)
    means = [t_mean, d_mean]
    lows  = [t_mean - t_lo, d_mean - d_lo]
    highs = [t_hi - t_mean, d_hi - d_mean]
    ax.bar(x, means)
    ax.errorbar(x, means, yerr=[lows, highs], fmt="none", capsize=6, linewidth=1.5)
    ax.axhline(0, color="k", linewidth=1, alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(["Treated","Defensive"])
    ax.set_ylabel("Mean Δ volatility (post − pre)")
    ax.set_title(f"Change in Conditional Volatility by Group\nEvent: {event_date}")
    out_fig = os.path.join(FIG_DIR, f"vol_vol_group_bars_{event_date}.png")
    plt.tight_layout(); plt.savefig(out_fig, dpi=150); plt.close()

    print(f"[OK] Saved group summary → {summary_path}")
    print(f"[OK] Saved figure        → {out_fig}")
    print("\n[Welch t-test on Δvol]")
    print(f"  t = {tstat:.3f}, p = {pval:.3f}")
    print("\n[Bootstrap 95% CIs for mean Δvol]")
    print(f"  Treated  : mean={t_mean:.4f}, CI=({t_lo:.4f}, {t_hi:.4f})")
    print(f"  Defensive: mean={d_mean:.4f}, CI=({d_lo:.4f}, {d_hi:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_date", type=str, required=True)
    args = ap.parse_args()
    main(args.event_date)
