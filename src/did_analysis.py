# src/did_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from datetime import datetime

# ---------- Config ----------
EVENT_DATE_STR = "2021-03-23"
EVENT_DATE = pd.to_datetime(EVENT_DATE_STR)

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(BASE, "../results/tables")
FIGS = os.path.join(BASE, "../results/figures")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

PANEL_PATH = os.path.join(TABLES, f"event_study_panel_{EVENT_DATE_STR}.csv")

# ---------- Helpers ----------
def pick(cols, cands, required=True):
    for c in cands:
        if c in cols:
            return c
    if required:
        raise KeyError(f"Missing expected column among {cands}. Available: {list(cols)}")
    return None

# ---------- Load panel (daily series in event window) ----------
df = pd.read_csv(PANEL_PATH)
# robust column detection
date_col   = pick(df.columns, ["date","Date"])
ticker_col = pick(df.columns, ["ticker","Ticker","symbol","Symbol"])
ar_col     = pick(df.columns, ["AR","ar","abnormal_return","AbnormalReturn"])
car_col    = pick(df.columns, ["CAR","car","cumulative_abnormal_return","CumulativeAbnormalReturn"], required=False)

# parse date
df[date_col] = pd.to_datetime(df[date_col])

# ---------- Map tickers -> sectors & treated/defensive ----------
TREATED_SECTORS = {"Metals","Energy","Autos","Infrastructure","Chemicals"}
DEFENSIVE_SECTORS = {"FMCG","Pharma","Utilities","Telecom","IT","Cement","Finance","Banks","Consumer"}

ticker_to_sector = {
    # Energy
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy","IOC.NS":"Energy","BPCL.NS":"Energy",
    # Metals
    "TATASTEEL.NS":"Metals","JSWSTEEL.NS":"Metals","HINDALCO.NS":"Metals","COALINDIA.NS":"Metals",
    # Autos
    "MARUTI.NS":"Autos","M&M.NS":"Autos","TATAMOTORS.NS":"Autos","BAJAJ-AUTO.NS":"Autos",
    "EICHERMOT.NS":"Autos","HEROMOTOCO.NS":"Autos",
    # Pharma
    "SUNPHARMA.NS":"Pharma","DRREDDY.NS":"Pharma","CIPLA.NS":"Pharma","DIVISLAB.NS":"Pharma",
    # FMCG / Consumer (defensive bucket)
    "HINDUNILVR.NS":"FMCG","ITC.NS":"FMCG","NESTLEIND.NS":"FMCG","BRITANNIA.NS":"FMCG",
    "TATACONSUM.NS":"FMCG","ASIANPAINT.NS":"FMCG","TITAN.NS":"Consumer",
    # IT
    "TCS.NS":"IT","INFY.NS":"IT","WIPRO.NS":"IT","HCLTECH.NS":"IT","TECHM.NS":"IT",
    # Telecom
    "BHARTIARTL.NS":"Telecom",
    # Cement / Materials (defensive bucket here)
    "ULTRACEMCO.NS":"Cement","SHREECEM.NS":"Cement","GRASIM.NS":"Cement",
    # Chemicals
    "UPL.NS":"Chemicals",
    # Infra / Industrials
    "ADANIPORTS.NS":"Infrastructure","LT.NS":"Infrastructure",
    # Utilities
    "NTPC.NS":"Utilities","POWERGRID.NS":"Utilities",
    # Finance / Banks / NBFCs
    "HDFCBANK.NS":"Finance","ICICIBANK.NS":"Finance","KOTAKBANK.NS":"Finance","AXISBANK.NS":"Finance",
    "SBIN.NS":"Finance","INDUSINDBK.NS":"Finance","BAJFINANCE.NS":"Finance","BAJAJFINSV.NS":"Finance",
}

df["sector"] = df[ticker_col].map(ticker_to_sector)
missing = df.loc[df["sector"].isna(), ticker_col].unique()
if len(missing) > 0:
    print("[WARN] Missing sector for:", missing)

def to_group(sector):
    if pd.isna(sector): return "Other"
    if sector in TREATED_SECTORS: return "Treated"
    if sector in DEFENSIVE_SECTORS: return "Defensive"
    return "Other"

df["Group"] = df["sector"].apply(to_group)
df = df[df["Group"].isin(["Treated","Defensive"])].copy()

# ---------- Build DiD indicators ----------
df["Post"] = (df[date_col] >= EVENT_DATE).astype(int)
df["TreatedFlag"] = (df["Group"] == "Treated").astype(int)

# Outcome: abnormal return (daily)
df["y"] = df[ar_col]

# Keep a reasonable symmetric window (optional). You can comment this out if panel already restricted.
rel_days = (df[date_col] - EVENT_DATE).dt.days
df = df[(rel_days >= -10) & (rel_days <= 20)].copy()
df["rel_day"] = rel_days[(rel_days >= -10) & (rel_days <= 20)]

# ---------- DiD with two-way fixed effects ----------
# AR_it ~ Treated_i*Post_t + stock FE + date FE
formula = f"y ~ TreatedFlag*Post + C({ticker_col}) + C({date_col})"
model = smf.ols(formula, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df[ticker_col]}
)
did_coef = model.params.get("TreatedFlag:Post", np.nan)
did_se   = model.bse.get("TreatedFlag:Post", np.nan)
did_p    = model.pvalues.get("TreatedFlag:Post", np.nan)

print("\n=== Difference-in-Differences (Two-way FE, cluster by ticker) ===")
print(model.summary().tables[1])
print(f"\n[Key] DiD coefficient (Treated x Post) = {did_coef:.6f}  (SE {did_se:.6f}, p={did_p:.3f})")
print("Interpretation: daily AR difference for treated vs defensive AFTER event, net of stock & date effects.\n")

# ---------- Save regression table ----------
out_csv = os.path.join(TABLES, f"did_summary_{EVENT_DATE_STR}.csv")
pd.DataFrame({
    "term": model.params.index,
    "coef": model.params.values,
    "std_err": model.bse.values,
    "pvalue": model.pvalues.values
}).to_csv(out_csv, index=False)
print(f"[OK] Saved DiD summary → {out_csv}")

# ---------- Visualization: pre-trends & post gap ----------
# Group-mean AR by relative day
g = df.groupby(["rel_day","Group"])["y"].mean().reset_index()
pivot = g.pivot(index="rel_day", columns="Group", values="y").sort_index()

plt.figure(figsize=(9,4.5))
plt.axvline(0, linestyle="--")
if "Treated" in pivot: plt.plot(pivot.index, pivot["Treated"]*100, label="Treated (mean AR)")
if "Defensive" in pivot: plt.plot(pivot.index, pivot["Defensive"]*100, label="Defensive (mean AR)")
plt.title("Event-Time Mean Abnormal Returns (pre/post)")
plt.xlabel("Days from event")
plt.ylabel("Mean AR (%)")
plt.legend()
plt.tight_layout()
fig1 = os.path.join(FIGS, f"did_event_time_means_{EVENT_DATE_STR}.png")
plt.savefig(fig1, dpi=300)
print(f"[OK] Saved figure → {fig1}")

# Difference curve: Treated minus Defensive
if {"Treated","Defensive"}.issubset(pivot.columns):
    diff = (pivot["Treated"] - pivot["Defensive"]) * 100
    plt.figure(figsize=(9,4.5))
    plt.axhline(0, color="gray", linewidth=1)
    plt.axvline(0, linestyle="--")
    plt.plot(diff.index, diff.values, label="Treated - Defensive (mean AR, pp)")
    plt.title("Event-Time Difference (Treated − Defensive)")
    plt.xlabel("Days from event")
    plt.ylabel("Percentage points")
    plt.legend()
    plt.tight_layout()
    fig2 = os.path.join(FIGS, f"did_event_time_diff_{EVENT_DATE_STR}.png")
    plt.savefig(fig2, dpi=300)
    print(f"[OK] Saved figure → {fig2}")
