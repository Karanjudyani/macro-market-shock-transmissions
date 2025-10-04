# Macro-Market-Shock-Transmissions

This project studies how **global shocks propagate into Indian equity markets**, using the **2021 Suez Canal blockage** as a case study.  
It applies an **event study methodology** across the NIFTY 50 index, analyzing stock- and sector-level responses to the disruption.  

---

## 🚀 Motivation
Global shocks — such as supply chain disruptions, commodity bottlenecks, or geopolitical conflicts — can ripple through financial markets.  
This project investigates:
- How Indian equity sectors react differently to such shocks (cyclical vs. defensive).
- How abnormal returns (AR) and cumulative abnormal returns (CAR) evolve in the short run.
- Whether market reactions align with economic intuition about trade and supply chain exposures.

This serves as:
- A **research demonstration** for economics/finance PhD applications.
- A **technical portfolio project** showcasing Python, data pipelines, and econometric modeling.

---

## 📊 Data
- **Equity Data**: NIFTY 50 constituents (Yahoo Finance, daily OHLCV).
- **Market Benchmark**: NIFTY 50 Index (`^NSEI`).
- **Commodities & FX**: Brent Crude (`BZ=F`), USD/INR (`INR=X`).
- **Volatility Indicators**: India VIX (`^INDIAVIX`), CBOE VIX (`^VIX`).

Date range: **Dec 2020 – Jun 2021** (covers estimation + event windows).

---

## 🔍 Methodology
### Event Study Setup
- **Estimation Window**: `[-120, -21]` trading days before event.  
  → Used to fit the market model:  
  \[
  r_{i,t} = \alpha_i + \beta_i r_{m,t} + \epsilon_{i,t}
  \]

- **Event Window**: `[-5, +20]` around event date (2021-03-23).  
  → Compute **Abnormal Returns (AR)** and **Cumulative Abnormal Returns (CAR)**.

- **Outputs**:
  - Stock-level CAR at +5 and +10 days.
  - Sector-level aggregations (mean, median, boxplots).
  - Treated (cyclical) vs Defensive sector comparisons.

### Workflow
1. **Data Pipeline** (`src/data_download.py`)  
   Downloads and merges Yahoo Finance data.
2. **Event Study Engine** (`src/event_study.py`)  
   Runs estimation regressions, computes AR & CAR.
3. **Sector Analysis** (`src/sector_analysis.py`)  
   Aggregates CARs by sector; produces figures and tables.

---

## 📈 Results
### Stock-level
- Mixed reactions across stocks: some cyclical sectors showed **short-term positive abnormal returns** despite expectations of disruption.
- Defensive sectors remained largely insulated.

### Sector-level
- Cyclical (e.g. Autos, Metals, Energy) vs Defensive (e.g. FMCG, Pharma, IT) patterns visible.  
- Evidence of **transitory positive market adjustment** around the event date.

*Figures available in `results/figures/`.*

## 🧑‍💻 Author
Karan Judyani  
- Interests: International Macro, Asset Pricing, Quantitative Finance.    

---

## 🛠️ How to Run
Clone repo, set up environment, and run the full pipeline:

```bash
git clone https://github.com/Karanjudyani/macro-market-shock-transmissions.git
cd macro-market-shock-transmissions
pip install -r requirements.txt

# Run full pipeline
python src/data_download.py --start 2020-12-01 --end 2021-06-30 && \
python src/event_study.py --event_date 2021-03-23 --pre_days 120 --post_days 20 --car_k1 5 --car_k2 10 && \
python src/sector_analysis.py --event_date 2021-03-23


