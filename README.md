# 🌍 Macro-Market-Shock-Transmissions

This project studies how **global shocks propagate into Indian equity markets**, using the **2021 Suez Canal blockage** as a case study.  
It applies **event study, difference-in-differences, and volatility transmission models** across the NIFTY 50 index to analyze stock- and sector-level responses to the disruption and its resolution.  

---

## 🚀 Motivation

Global events such as supply chain disruptions, commodity bottlenecks, and geopolitical conflicts can ripple through financial markets in unexpected ways.  
The 2021 *Ever Given* Suez Canal blockage was one such episode — halting nearly 12% of global trade and shaking global oil and shipping prices.

This project investigates:
- How Indian equity sectors reacted differently to the **blockage** and the **resolution**.
- Whether **cyclical sectors** (Energy, Industrials, Metals) responded more strongly than **defensives** (Pharma, IT, FMCG).
- How **volatility and global risk linkages** behaved during and after the event.

It serves as:
- A **research demonstration** for macro-finance and econometrics.  
- A **portfolio project** showcasing Python-based event analysis, econometric modeling, and visualization.

---

## 🧾 Summary of Findings

| Event Phase | Main Observation | Interpretation |
|--------------|------------------|----------------|
| **Blockage (Mar 23, 2021)** | Mild rise in volatility among cyclical sectors, no significant return abnormality | Market viewed the blockage as temporary — limited contagion |
| **Resolution (Apr 3, 2021)** | Significant volatility *decline* in Energy & Financials, correlated with falling Brent & VIX | Evidence of **global spillover** — stabilization transmitted to India |
| **Overall** | Transitory disruption, rapid normalization | Indian markets reflected **re-anchoring**, not crisis contagion |

---

## 📊 Data

- **Equity Data:** NIFTY 50 constituents (Yahoo Finance OHLCV)
- **Market Benchmark:** NIFTY 50 Index (`^NSEI`)
- **Commodities & FX:** Brent Crude (`BZ=F`), USD/INR (`INR=X`)
- **Volatility Indicators:** India VIX (`^INDIAVIX`), CBOE VIX (`^VIX`)
- **Date Range:** Dec 2020 – Jun 2021 (estimation + event windows)

Sector mapping provided in `data/ticker_sectors.csv`.

---

## 🧮 Methodology

### 🧠 1. Event Study
- **Estimation Window:** `[-120, -21]`  
  \[
  r_{i,t} = \alpha_i + \beta_i r_{m,t} + \epsilon_{i,t}
  \]
- **Event Windows:**  
  - Blockage: ±5 to ±20 days around *2021-03-23*  
  - Resolution: ±5 to ±20 days around *2021-04-03*
- **Outputs:**  
  Abnormal Returns (AR), Cumulative Abnormal Returns (CAR₅, CAR₁₀), and sector-level bootstrapped confidence intervals.

---

### ⚖️ 2. Difference-in-Differences (DiD / DDD)
Compared:
- **Treated** = Cyclical sectors (Energy, Metals, Industrials, Autos)  
- **Control** = Defensive sectors (Pharma, FMCG, IT, Utilities)

Regression setup:
\[
y_{it} = \alpha + \beta_1 \text{Post}_t + \beta_2 \text{Treated}_i + \beta_3 (\text{Post}_t \times \text{Treated}_i) + \epsilon_{it}
\]
Extended (DDD) model adds interaction for high exposure (e.g., trade-linked or energy-dependent firms).

🧩 **Insight:**  
- DiD estimates insignificant → no systematic pricing contagion.  
- Market response concentrated in volatility, not in mean returns.

---

### 📈 3. Volatility Modeling
- Computed pre- and post-event conditional volatilities (σ) using GARCH(1,1).  
- Δσ = σ_post − σ_pre at stock level.
- Aggregated by sector and exposure group.

**Results:**  
- March 23 → volatility spike limited to Energy, Metals, Autos.  
- April 3 → volatility normalization across all sectors.

---

### 📊 4. Volatility Group Comparison
| Event | Treated Δσ (median) | Defensive Δσ (median) | Interpretation |
|--------|----------------------|------------------------|----------------|
| Mar 23 | +0.009 | ≈ 0 | Temporary spike in cyclicals |
| Apr 03 | ≈ 0 | ≈ 0 | Full mean reversion after refloating |

---

### 🌐 5. Global Linkages Regression
Linked sector-level volatility changes to global macro shocks:

\[
\Delta\sigma_i = \alpha + \beta_1 \text{Treated}_i + \beta_2 (\text{EnergyExp}_i \times \Delta \text{Brent}) + \beta_3 (\text{RiskExp}_i \times \Delta \text{VIX}) + \beta_4 (\text{FXExp}_i \times \Delta \text{INR}) + \epsilon_i
\]

#### 🔹 Blockage Phase (Mar 23)
- ΔBrent = –3.7%, ΔVIX = +0.5%, ΔINR ≈ 0  
- R² = 0.08 → weak explanatory power  
➡ No significant spillover → limited contagion.

#### 🔹 Resolution Phase (Apr 3)
- ΔBrent = –2.5%, ΔVIX = –8.0%, ΔINR = +1.1%  
- R² = 0.25 → moderate explanatory power  
➡ Significant coefficients:
  - `EnergyExp × ΔBrent` = –0.22 (p = 0.008)  
  - `RiskExp × ΔVIX` = +0.07 (p = 0.009)  

🧠 **Interpretation:**  
After the canal reopened, **global risk compression** (falling oil & VIX) transmitted strongly to Indian markets.  
Energy and Financial sectors’ volatilities declined, mirroring the global calm.  
This marks a **measurable spillover** — the event’s resolution *re-anchored* global commodity and risk-price channels to Indian equities.

---

## 📜 Discussion

During the blockage week, investors viewed the shock as short-term. Indian markets remained largely stable, with only minor volatility upticks in supply-sensitive industries.  
Once the crisis resolved, global variables (Brent, VIX) began influencing Indian volatility again — evidence of **restored international linkages**.

> In simple terms: when the world calmed down, India’s cyclicals calmed down too.

This asymmetric response suggests that emerging markets like India are **less sensitive to the onset of external stress**, but **quickly re-integrate** once uncertainty subsides.

---

## 🧩 Conclusion

- The Suez blockage created a **short-lived volatility shock**, not a prolonged crisis.  
- **During the event:** limited transmission — markets remained resilient.  
- **After resolution:** significant global spillovers — volatility declined with falling oil and risk levels.  
- Highlights India’s **asymmetric global linkage** — partial decoupling in stress, strong integration in recovery.

---

## 🔮 Next Steps

1. Add firm-level controls (size, liquidity, beta).  
2. Test spillovers on BSE Midcap and sector ETFs.  
3. Extend to **VAR-based dynamic models**.  
4. Compare with **Red Sea 2023 shipping blockade** as a modern replication.  

---

## 🧠 Tech Stack
- **Language:** Python (3.11)
- **Libraries:** pandas, numpy, statsmodels, yfinance, arch, matplotlib, seaborn
- **Workflow:** Modular scripts under `/src`:
  - `data_download.py` — data ingestion  
  - `event_study.py` — AR/CAR estimation  
  - `did_analysis.py` — difference-in-differences modeling  
  - `volatility_models.py` — GARCH volatility computation  
  - `global_linkages.py` — international spillover regression  

All output figures and tables are stored in `/results/figures/` and `/results/tables/`.

---

## 🛠️ How to Run

```bash
git clone https://github.com/Karanjudyani/macro-market-shock-transmissions.git
cd macro-market-shock-transmissions
pip install -r requirements.txt

# 1️⃣ Download and preprocess data
python src/data_download.py --start 2020-12-01 --end 2021-06-30

# 2️⃣ Run event studies
python src/event_study.py --event_date 2021-03-23 --pre_days 120 --post_days 20 --car_k1 5 --car_k2 10
python src/event_study.py --event_date 2021-04-03 --pre_days 120 --post_days 20 --car_k1 5 --car_k2 10

# 3️⃣ Run difference-in-differences
python src/did_analysis.py --event_date 2021-03-23
python src/did_analysis.py --event_date 2021-04-03

# 4️⃣ Compute volatility and sector comparisons
python src/volatility_models.py --event_date 2021-03-23
python src/volatility_models.py --event_date 2021-04-03
python src/volatility_group_comparison.py --event_date 2021-03-23
python src/volatility_group_comparison.py --event_date 2021-04-03

# 5️⃣ Global linkage regressions
python src/global_linkages.py --event_date 2021-03-23
python src/global_linkages.py --event_date 2021-04-03
