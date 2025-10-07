import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# ==============================
# 1️⃣ PATHS & CONFIG
# ==============================
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "../results/tables")
FIGS = os.path.join(BASE, "../results/figures")

EVENT_DATE = "2021-03-23"

# ==============================
# 2️⃣ LOAD EVENT STUDY RESULTS
# ==============================
summary_path = os.path.join(RESULTS, f"event_study_summary_{EVENT_DATE}.csv")
summary = pd.read_csv(summary_path)

print(f"[INFO] Loaded summary: {summary.shape[0]} rows, {summary.columns.tolist()}")

# --- Robust column detection ---
car5_candidates = ["CAR5", "car5", "CAR_5d", "car_5d"]
car10_candidates = ["CAR10", "car10", "CAR_10d", "car_10d"]

car5_col = next((c for c in car5_candidates if c in summary.columns), None)
car10_col = next((c for c in car10_candidates if c in summary.columns), None)

if not car5_col or not car10_col:
    raise KeyError(f"Could not find CAR columns. Available: {summary.columns.tolist()}")

print(f"[INFO] Using columns: {car5_col}, {car10_col}")

# ==============================
# 3️⃣ DEFINE CUSTOM SECTOR MAPPING
# ==============================
TREATED_SECTORS = {"Metals","Energy","Autos","Infrastructure","Chemicals"}
DEFENSIVE_SECTORS = {"FMCG","Pharma","Utilities","Telecom","IT","Cement","Finance","Banks","Consumer"}

ticker_to_sector = {
    # Energy / Oil & Gas
    "RELIANCE.NS": "Energy",
    "ONGC.NS": "Energy",
    "IOC.NS": "Energy",
    "BPCL.NS": "Energy",

    # Metals / Mining / Materials
    "TATASTEEL.NS": "Metals",
    "JSWSTEEL.NS": "Metals",
    "HINDALCO.NS": "Metals",
    "COALINDIA.NS": "Metals",

    # Autos (4-wheelers, 2-wheelers, CVs)
    "MARUTI.NS": "Autos",
    "M&M.NS": "Autos",
    "TATAMOTORS.NS": "Autos",
    "BAJAJ-AUTO.NS": "Autos",
    "EICHERMOT.NS": "Autos",
    "HEROMOTOCO.NS": "Autos",

    # Pharma
    "SUNPHARMA.NS": "Pharma",
    "DRREDDY.NS": "Pharma",
    "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma",

    # FMCG / Consumer Staples & Discretionary (defensive bucket for this project)
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG",
    "TATACONSUM.NS": "FMCG",
    "ASIANPAINT.NS": "FMCG",
    "TITAN.NS": "Consumer",   # treating as Defensive for this study

    # IT / Tech Services
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "WIPRO.NS": "IT",
    "HCLTECH.NS": "IT",
    "TECHM.NS": "IT",

    # Telecom
    "BHARTIARTL.NS": "Telecom",

    # Cement / Building Materials (defensive bucket)
    "ULTRACEMCO.NS": "Cement",
    "SHREECEM.NS": "Cement",
    "GRASIM.NS": "Cement",

    # Chemicals
    "UPL.NS": "Chemicals",

    # Infrastructure / Transport / Industrials
    "ADANIPORTS.NS": "Infrastructure",
    "LT.NS": "Infrastructure",

    # Utilities (power generation / transmission)
    "NTPC.NS": "Utilities",
    "POWERGRID.NS": "Utilities",

    # Finance / Banks / NBFCs (treat as Defensive for this study)
    "HDFCBANK.NS": "Finance",
    "ICICIBANK.NS": "Finance",
    "KOTAKBANK.NS": "Finance",
    "AXISBANK.NS": "Finance",
    "SBIN.NS": "Finance",
    "INDUSINDBK.NS": "Finance",
    "BAJFINANCE.NS": "Finance",
    "BAJAJFINSV.NS": "Finance",
}


summary["sector"] = summary["ticker"].map(ticker_to_sector)

missing = summary.loc[summary["sector"].isna(), "ticker"].unique()
if len(missing) > 0:
    print(f"[WARN] These tickers have no assigned sector: {missing}")

# ==============================
# 4️⃣ GROUP INTO TREATED VS DEFENSIVE
# ==============================
def classify_group(sector_name: str) -> str:
    if pd.isna(sector_name):
        return "Other"
    if sector_name in TREATED_SECTORS:
        return "Treated"
    elif sector_name in DEFENSIVE_SECTORS:
        return "Defensive"
    else:
        return "Other"

summary["Group"] = summary["sector"].apply(classify_group)
summary = summary[summary["Group"].isin(["Treated", "Defensive"])].copy()

print(f"[INFO] Group counts:\n{summary['Group'].value_counts()}")

# ==============================
# 5️⃣ STATISTICAL TESTS (Welch t-test)
# ==============================
def welch_t(a, b):
    a = pd.Series(a).dropna().values
    b = pd.Series(b).dropna().values
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return t_stat, p_val, np.mean(a), np.mean(b)

print("\n[Welch’s t-tests for mean CARs]\n")
for col in [car5_col, car10_col]:
    t_stat, p_val, mean_treat, mean_def = welch_t(
        summary.loc[summary["Group"]=="Treated", col],
        summary.loc[summary["Group"]=="Defensive", col]
    )
    print(f"{col}: t={t_stat:.3f}, p={p_val:.3f}, mean_treated={mean_treat:.4f}, mean_defensive={mean_def:.4f}")

# ==============================
# 6️⃣ BOOTSTRAP CONFIDENCE INTERVALS
# ==============================
def bootstrap_ci(data, n_boot=2000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.Series(data).dropna().values
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    means = [rng.choice(data, len(data), replace=True).mean() for _ in range(n_boot)]
    lower = np.percentile(means, (100-ci)/2)
    upper = np.percentile(means, 100 - (100-ci)/2)
    return np.mean(data), lower, upper

rows = []
for grp in ["Treated", "Defensive"]:
    for metric in [car5_col, car10_col]:
        vals = summary.loc[summary["Group"]==grp, metric]
        mean, low, high = bootstrap_ci(vals, n_boot=3000, ci=95)
        rows.append([grp, metric, mean, low, high, len(vals)])

boot_df = pd.DataFrame(rows, columns=["Group","Metric","Mean","Low_CI","High_CI","N"])
boot_out = os.path.join(RESULTS, f"bootstrap_ci_{EVENT_DATE}.csv")
boot_df.to_csv(boot_out, index=False)
print(f"\n[OK] Saved bootstrap CI table → {boot_out}")

# ==============================
# 7️⃣ PLOT BOOTSTRAP CIs
# ==============================
os.makedirs(FIGS, exist_ok=True)
scale = 100.0
order = ["Treated", "Defensive"]
metrics = [car5_col, car10_col]

fig, ax = plt.subplots(figsize=(7,5))
width = 0.35
x0 = np.arange(len(order))

for i, metric in enumerate(metrics):
    sub = boot_df[boot_df["Metric"]==metric].set_index("Group").loc[order]
    centers = x0 + (-width/2 if i==0 else width/2)
    ax.bar(centers, sub["Mean"]*scale, width=width, label=metric)
    for j, g in enumerate(order):
        mean = sub.loc[g,"Mean"]*scale
        low = sub.loc[g,"Low_CI"]*scale
        high = sub.loc[g,"High_CI"]*scale
        ax.errorbar(centers[j], mean,
                    yerr=[[mean - low],[high - mean]],
                    fmt='none', ecolor='k', capsize=4)

ax.set_xticks(x0)
ax.set_xticklabels(order)
ax.set_ylabel("Mean CAR (%), 95% CI")
ax.set_title("Bootstrap Confidence Intervals for Mean CARs")
ax.legend()
plt.tight_layout()

fig_path = os.path.join(FIGS, f"bootstrap_CI_CARs_{EVENT_DATE}.png")
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"[OK] Saved figure → {fig_path}")
