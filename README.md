# AI/IoT Smart Hotel Housekeeping Management System
### Dual-Paper Research Architecture & Operational Optimization Framework

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Google OR-Tools](https://img.shields.io/badge/Optimization-OR--Tools%20CP--SAT-green.svg)](https://developers.google.com/optimization)

An end-to-end research-grade engineering system designed to optimize hospitality operations, guest satisfaction, techno-economic feasibility, environmental sustainability, and technology adoption.

The project partitions the five foundational research objectives into **two independent journal papers** formatted for Tier-1 peer-reviewed publications:
* **Paper 1 (Operations & Techno-Economics)**: *Predictive-Prescriptive Optimization for Smart Hospitality Operations: Coupling Gradient-Boosted Demand Forecasting, Constraint-Satisfaction Task Scheduling, and Monte Carlo Cost Modeling* (Targeting *IJHM* / *Computers & Industrial Engineering*).
* **Paper 2 (Guest Experience, Sustainability & TAM)**: *Data-Driven Eco-Hospitality: Aspect-Based Sentiment Mining, Personalized Housekeeping Profiles, and Calibrated Resource-Carbon Footprint Reduction under Adoption Barriers* (Targeting *Journal of Cleaner Production* / *Tourism Management*).

---

## 📚 Documentation & Developer Guides

* **[Comprehensive Developer Documentation (`docs/DEVELOPER_GUIDE.md`)](file:///home/pranav/HotelManagement_Project/docs/DEVELOPER_GUIDE.md)**: Full component-by-component walkthrough, architectural rationale, code explanations, and extension guidelines.
* **[Concepts, Methodologies & Terminologies Primer (`docs/METHODOLOGY_AND_TERMINOLOGY.md`)](file:///home/pranav/HotelManagement_Project/docs/METHODOLOGY_AND_TERMINOLOGY.md)**: Beginner-to-advanced explanation of all mathematical formulas, machine learning concepts, operations research algorithms, building physics, and survey psychometrics.
* **[Paper 1 Academic Briefing (`reports/Paper1_Operations_and_Economics_Briefing.md`)](file:///home/pranav/HotelManagement_Project/reports/Paper1_Operations_and_Economics_Briefing.md)**: Complete manuscript draft for the operational and financial stream.
* **[Paper 2 Academic Briefing (`reports/Paper2_Guest_Sustainability_and_TAM_Briefing.md`)](file:///home/pranav/HotelManagement_Project/reports/Paper2_Guest_Sustainability_and_TAM_Briefing.md)**: Complete manuscript draft for the guest experience, sustainability, and TAM adoption stream.

---

## ⚡ Quickstart: Running the System

### 1. Activate Environment
The virtual environment is already provisioned with all 61 required packages:
```bash
source .venv/bin/activate
```

### 2. Execute Full Experiment Pipeline
Run the entire end-to-end data processing, model training, discrete optimization, Monte Carlo simulation, and 300 DPI figure generation pipeline:
```bash
python run_pipeline.py
```
*Runtime: ~90 seconds. All benchmark tables and publication-grade figures will be generated automatically in `results/paper1/` and `results/paper2/`.*

### 3. Launch the Interactive Decision Dashboard
Start the multi-page Streamlit application:
```bash
streamlit run src/dashboard/app.py
```
Open `http://localhost:8501` to explore:
* Out-of-time forecasting trajectories and operational boxplots.
* Interactive Monte Carlo sliders for occupancy, wages, and IoT CapEx with instant payback recalculation.
* Aspect sentiment radar plots and guest cluster profiles.
* Calibrated carbon offset calculator across Indian CEA and US EPA grids.
* Publication Center with **ready-to-copy LaTeX code** for all 11 tables.

---

## 📊 Summary of Empirical Results

### Paper 1: Operations & Techno-Economics
1. **Workload Forecasting**: Tabular regression on 32 engineered lag/lead features achieves $R^2 = 0.9269$ (Ridge) and $R^2 = 0.8859$ (LightGBM) under a strict chronological 2017 test partition, outperforming the moving average baseline ($R^2 = 0.7823$).
2. **Floor Scheduling Optimization**: Google OR-Tools CP-SAT discrete constraint optimizer eliminates **34.03% of shift overtime** ($p = 2.79 \times 10^{-60}$) and reduces late room check-in SLA breaches by **77.15%** ($p = 2.38 \times 10^{-7}$).
3. **Financial Feasibility**: 10,000-run Monte Carlo simulation demonstrates a baseline CPOR of ₹128.80 reduced to ₹124.30 under optimization, delivering an amortized **capital payback horizon of 16.4 months** for a 120-room property.

### Paper 2: Guest Experience, Sustainability & Adoption
1. **Aspect-Based Sentiment Mining**: Supervised TF-IDF + LinearSVC achieves **85.68% accuracy** and **0.6844 Macro-F1** across 14.9k housekeeping reviews (+21.4% improvement over lexicon rules).
2. **Guest Personalization**: $K$-Means clustering ($K=4$, Silhouette $= 0.418$) segments guests into *Eco-Conscious Minimalist* (17.1%), *Business Short-Stay* (34.0%), *Family High-Intensity* (26.6%), and *Luxury High-Touch* (22.2%) with tailored cleaning rules.
3. **Resource & Carbon Abatement**: Calibrated against NYC LL84 and BEE India standards, smart housekeeping saves **70,578 kWh of electricity** (12.1% reduction) and **476,824 L of water** annually, avoiding **51.40 Metric Tonnes of $\text{CO}_2\text{e}$** under the Indian power grid.
4. **Adoption Governance**: Empirical TAM survey of 32 hospitality executives reveals *Staff Digital Literacy Gaps* ($\text{RII} = 0.862$) and *Upfront IoT CapEx* ($\text{RII} = 0.844$) as top roadblocks, paired with actionable managerial mitigations.

---

## 🏛️ Project Structure

```
├── data/
│   ├── calibration/standards.json     # ISSA, FHRAI, BEE, CEA benchmarks
│   ├── raw/                           # Raw downloaded bookings & reviews
│   └── processed/                     # Cleaned parquet/CSV data files
├── docs/                              # Developer guide & terminology primer
├── reports/                           # Academic briefings for Paper 1 & Paper 2
├── results/
│   ├── paper1/                        # Tables, figures (PNG/PDF), metrics for Paper 1
│   └── paper2/                        # Tables, figures (PNG/PDF), metrics for Paper 2
├── src/
│   ├── data_pipeline/fetch_data.py    # Automated data acquisition & cleaning
│   ├── paper1_operations/             # Forecasting, OR-Tools scheduler, Monte Carlo
│   ├── paper2_sustainability/         # Aspect sentiment, energy ML, TAM survey
│   ├── visualization/generate_figures.py # 300 DPI IEEE/Elsevier figure generator
│   └── dashboard/app.py               # Streamlit multi-page research interface
├── requirements.txt                   # Pinned reproducible dependencies
└── run_pipeline.py                    # Master CLI orchestration pipeline
```
