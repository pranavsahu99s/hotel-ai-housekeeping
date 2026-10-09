# Developer Documentation & Implementation Guide
## AI/IoT Smart Hotel Housekeeping Management System

---

### Table of Contents
1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Dual-Paper Publication Framework](#2-dual-paper-publication-framework)
3. [Repository Directory Layout](#3-repository-directory-layout)
4. [Step-by-Step Component Walkthrough](#4-step-by-step-component-walkthrough)
   - [Component 1: Empirical Calibration (`standards.json`)](#component-1-empirical-calibration-standardsjson)
   - [Component 2: Data Ingestion & Pipeline (`fetch_data.py`)](#component-2-data-ingestion--pipeline-fetch_datapy)
   - [Component 3: Demand Forecasting (`forecasting.py`)](#component-3-demand-forecasting-forecastingpy)
   - [Component 4: Discrete Floor Scheduling (`scheduler.py`)](#component-4-discrete-floor-scheduling-schedulerpy)
   - [Component 5: Monte Carlo Cost Engine (`cost_monte_carlo.py`)](#component-5-monte-carlo-cost-engine-cost_monte_carlopy)
   - [Component 6: NLP Sentiment & Personalization (`nlp_sentiment.py`)](#component-6-nlp-sentiment--personalization-nlp_sentimentpy)
   - [Component 7: Calibrated Environmental ML (`environmental_models.py`)](#component-7-calibrated-environmental-ml-environmental_modelspy)
   - [Component 8: TAM Survey & Barrier Analysis (`tam_survey_analysis.py`)](#component-8-tam-survey--barrier-analysis-tam_survey_analysispy)
   - [Component 9: Publication Figures (`generate_figures.py`)](#component-9-publication-figures-generate_figurespy)
   - [Component 10: Streamlit Research Dashboard (`app.py`)](#component-10-streamlit-research-dashboard-apppy)
   - [Component 11: Master Orchestration (`run_pipeline.py`)](#component-11-master-orchestration-run_pipelinepy)
5. [How to Run, Test, and Reproduce](#5-how-to-run-test-and-reproduce)
6. [Developer Extension Guide](#6-developer-extension-guide)

---

### 1. Project Overview & Architecture

#### Motivation
Hotel housekeeping operations represent a high-stakes operational domain where physical constraints (room availability, staff shift hours, elevator transit delays), guest sentiment (cleanliness and hygiene expectations), financial expenses (labor overtime, chemical costs), and environmental footprints (laundry water, HVAC energy) intersect.

#### Architectural Philosophy: Four Modular Subsystems
Rather than attempting to train a single monolithic neural network, this system is engineered as **four independent, swappable analytical modules** reporting into a **shared evaluation layer**:
1. **Predictive Workload Engine**: Forecasts checkout turnover volume using tabular gradient boosting.
2. **Prescriptive Operations Engine**: Optimizes room assignments and floor schedules using discrete constraint satisfaction.
3. **Guest NLP & Personalization Engine**: Mines aspect sentiment from unstructured reviews and derives behavioral guest profiles.
4. **Calibrated Resource & Abatement Engine**: Models physical building energy/water savings and converts them into policy-grade carbon equivalents.

```
                              DATA & EXECUTION PIPELINE
                              
       [119,390 Bookings]          [20,491 Reviews]          [NYC LL84 / BEE Energy]
               │                          │                            │
               ▼                          ▼                            ▼
      [fetch_data.py]            [fetch_data.py]              [fetch_data.py]
               │                          │                            │
               ├──────────────────────────┼────────────────────────────┤
               │                          │                            │
               ▼                          ▼                            ▼
        [PAPER 1 ENGINE]           [PAPER 2 ENGINE]           [PAPER 2 ENGINE]
    ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
    │ forecasting.py       │   │ nlp_sentiment.py     │   │ environmental_models │
    │ scheduler.py         │   │                      │   │ tam_survey_analysis  │
    │ cost_monte_carlo.py  │   │                      │   │                      │
    └──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
               │                          │                          │
               └──────────────────────────┼──────────────────────────┘
                                          │
                                          ▼
                             ┌──────────────────────────┐
                             │ generate_figures.py      │
                             │ (300 DPI PNG & PDF)      │
                             └────────────┬─────────────┘
                                          │
                                          ▼
                             ┌──────────────────────────┐
                             │ src/dashboard/app.py     │
                             │ (Streamlit Dashboard)    │
                             └──────────────────────────┘
```

---

### 2. Dual-Paper Publication Framework

To maximize research visibility, the five foundational objectives are partitioned into two standalone journal papers:

#### Paper 1: Operational Efficiency & Techno-Economic Feasibility
* **Core Topics**: Objective 1 (Operational Efficiency & Scheduling) + Objective 3 (Cost-Effectiveness & Monte Carlo Feasibility).
* **Target Journals**: *International Journal of Hospitality Management* (Elsevier, Q1, IF 9.0), *Computers & Industrial Engineering* (Elsevier, Q1, IF 6.7).
* **Research Focus**: Predictive workload forecasting under chronological splits, discrete constraint scheduling avoiding floor transit overhead, and stochastic 10,000-run Monte Carlo financial analysis.

#### Paper 2: Guest Experience, Sustainability & Technology Adoption
* **Core Topics**: Objective 2 (Guest Satisfaction & Personalization) + Objective 4 (Environmental & Carbon Abatement) + Objective 5 (TAM/UTAUT Adoption Barriers).
* **Target Journals**: *Journal of Cleaner Production* (Elsevier, Q1, IF 9.7), *Tourism Management* (Elsevier, Q1, IF 10.9).
* **Research Focus**: Housekeeping Aspect-Based Sentiment Analysis (ABSA), $K$-Means guest clustering, ASHRAE/NYC LL84 calibrated energy-water regressions, and empirical Technology Acceptance Model (TAM) psychometrics.

---

### 3. Repository Directory Layout

```
HotelManagement_Project/
├── .venv/                         # Python 3.11 isolated virtual environment
├── data/
│   ├── calibration/
│   │   └── standards.json         # Industry calibration standards (ISSA, FHRAI, BEE, CEA)
│   ├── raw/
│   │   ├── hotel_bookings_raw.csv # 119k bookings (Antonio et al., 2019)
│   │   └── tripadvisor_reviews_raw.csv # 20k reviews (TripAdvisor corpus)
│   └── processed/
│       ├── daily_turnover_workload.csv  # 806 daily aggregated cleaning workloads
│       ├── hotel_bookings_cleaned.csv   # Cleaned reservation records with stay dates
│       ├── hotel_energy_benchmarks.csv  # 250 calibrated lodging properties (NYC LL84)
│       ├── hotel_reviews_tagged.csv     # 14.9k tagged housekeeping reviews
│       └── tam_adoption_survey.csv      # 32 primary executive survey responses
├── docs/
│   ├── DEVELOPER_GUIDE.md               # This master developer documentation
│   └── METHODOLOGY_AND_TERMINOLOGY.md   # Concepts and terminology primer
├── reports/
│   ├── Paper1_Operations_and_Economics_Briefing.md # Comprehensive manuscript report for Paper 1
│   └── Paper2_Guest_Sustainability_and_TAM_Briefing.md # Comprehensive manuscript report for Paper 2
├── results/
│   ├── paper1/
│   │   ├── figures/               # Figure 1, 2, 3 (300 DPI PNG and PDF)
│   │   ├── tables/                # Tables 1, 2, 3 in CSV format
│   │   └── *.json                 # Raw experimental metric ledgers
│   └── paper2/
│       ├── figures/               # Figure 4, 5, 6 (300 DPI PNG and PDF)
│       ├── tables/                # Tables 4 through 11 in CSV format
│       └── *.json                 # Raw experimental metric ledgers
├── src/
│   ├── data_pipeline/
│   │   └── fetch_data.py          # Data ingestion, cleaning, and synthetic calibration generator
│   ├── paper1_operations/
│   │   ├── forecasting.py         # Workload forecasting (LightGBM vs. Ridge vs. Moving Avg)
│   │   ├── scheduler.py           # SimPy floor simulation & OR-Tools CP-SAT discrete optimizer
│   │   └── cost_monte_carlo.py    # 10,000-run Monte Carlo stochastic financial engine
│   ├── paper2_sustainability/
│   │   ├── nlp_sentiment.py       # Aspect sentiment mining, K-Means clustering, rule matrix
│   │   ├── environmental_models.py# Calibrated energy/water ML regressions & carbon accounting
│   │   └── tam_survey_analysis.py # TAM psychometrics, Cronbach alpha, barrier RII ranking
│   ├── visualization/
│   │   └── generate_figures.py    # Publication figure generator (IEEE/Elsevier formatting)
│   └── dashboard/
│       └── app.py                 # Streamlit multi-page interactive research application
├── requirements.txt               # Pinned reproducible dependencies
└── run_pipeline.py                # Master CLI orchestration pipeline
```

---

### 4. Step-by-Step Component Walkthrough

#### Component 1: Empirical Calibration (`standards.json`)
* **Path**: [`data/calibration/standards.json`](file:///home/pranav/HotelManagement_Project/data/calibration/standards.json)
* **What it does**: Stores empirical baseline numbers derived from published industry standards:
  * **Cleaning Durations (ISSA 540 & AHLA/HFTP)**:
    * Standard stayover clean: $\mu = 22.0$ min ($\sigma = 3.5$).
    * Standard checkout clean: $\mu = 38.0$ min ($\sigma = 5.0$).
    * Suite checkout clean: $\mu = 55.0$ min ($\sigma = 7.0$).
  * **Labor Wages (FHRAI India & US BLS)**:
    * Median Indian housekeeper wage: ₹140.00 / hr (1.5x overtime multiplier).
    * Median US housekeeper wage: $18.50 / hr.
  * **Energy & Carbon Baselines (BEE India & CEA)**:
    * Baseline Hotel Energy Intensity: 18.5 kWh / room-night.
    * Baseline Hotel Water Intensity: 350.0 Liters / room-night.
    * CEA Grid Emission Factor: 0.716 kg $\text{CO}_2\text{e}$ / kWh.

---

#### Component 2: Data Ingestion & Pipeline (`fetch_data.py`)
* **Path**: [`src/data_pipeline/fetch_data.py`](file:///home/pranav/HotelManagement_Project/src/data_pipeline/fetch_data.py)
* **What it does**:
  1. Checks if raw files exist; if missing, automatically downloads `hotels.csv` and `tripadvisor_hotel_reviews.csv` via high-speed curl.
  2. Constructs exact `arrival_date` and `departure_date`.
  3. Uses fast vectorized `itertuples()` expansion to compute daily active stayovers between check-in and checkout.
  4. Tags reviews containing 20+ specific housekeeping keywords (`clean`, `bathroom`, `towel`, `smell`, `stain`, `linen`, etc.).
  5. Generates the NYC LL84/ASHRAE lodging energy benchmark dataset (250 properties) and TAM survey dataset (32 respondents).

---

#### Component 3: Demand Forecasting (`forecasting.py`)
* **Path**: [`src/paper1_operations/forecasting.py`](file:///home/pranav/HotelManagement_Project/src/paper1_operations/forecasting.py)
* **What it does**:
  1. Builds 32 tabular features including autoregressive lags ($t-1, t-7, t-14, t-28$), rolling window statistics (7D mean, 7D standard deviation), and calendar features.
  2. Enforces a strict chronological train/test split:
     * **Train**: 2015-07 to 2016-12 (521 days).
     * **Test**: 2017-01 to 2017-09 (257 days).
  3. Benchmarks four models:
     * *7-Day Moving Average* ($R^2 = 0.7823$, $\text{MAE} = 24.53$).
     * *Ridge Regression* ($R^2 = 0.9269$, $\text{MAE} = 17.65$).
     * *LightGBM Regressor* ($R^2 = 0.8859$, $\text{MAE} = 18.39$).
     * *Gradient Boosting Regressor* ($R^2 = 0.8833$, $\text{MAE} = 18.95$).
  4. Exports [Table 1](file:///home/pranav/HotelManagement_Project/results/paper1/tables/table1_forecasting_benchmark.csv) and feature importance rankings.

---

#### Component 4: Discrete Floor Scheduling (`scheduler.py`)
* **Path**: [`src/paper1_operations/scheduler.py`](file:///home/pranav/HotelManagement_Project/src/paper1_operations/scheduler.py)
* **What it does**:
  1. Generates 120 rooms across 4 floors operating with stochastic task durations sampled from ISSA 540 standards.
  2. Executes a **Greedy Round-Robin Baseline**: Assigns rooms to the staff member with the earliest availability.
  3. Formulates the **Google OR-Tools CP-SAT Discrete Optimizer**:
     * Creates binary decision variables $x_{t, s}$ linking each task $t$ to staff $s$.
     * Enforces `AddNoOverlap` so no housekeeper cleans multiple rooms simultaneously.
     * Incorporates penalties for shift overtime beyond 7 hours, SLA delays past 15:00 check-in, and floor transit overhead (3 mins per inter-floor transition).
  4. Simulates 30 consecutive operational shifts and computes paired $t$-tests:
     * **Overtime Incurred**: Reduced by **34.03%** ($p = 2.79 \times 10^{-60}$).
     * **SLA Breaches**: Reduced from 3.89% to 0.89% (**77.15% reduction**, $p = 2.38 \times 10^{-7}$).
     * **Staff Utilization**: Improved from 93.44% to 99.79% ($p = 1.59 \times 10^{-29}$).
  5. Exports [Table 2](file:///home/pranav/HotelManagement_Project/results/paper1/tables/table2_operational_scheduling_benchmark.csv).

---

#### Component 5: Monte Carlo Cost Engine (`cost_monte_carlo.py`)
* **Path**: [`src/paper1_operations/cost_monte_carlo.py`](file:///home/pranav/HotelManagement_Project/src/paper1_operations/cost_monte_carlo.py)
* **What it does**:
  1. Runs **10,000 stochastic iterations** representing independent operational years for a 120-room hotel.
  2. Samples hotel occupancy from a $\text{Beta}(14, 5)$ distribution and hourly wages from Gaussian distributions.
  3. Calculates Cost Per Occupied Room (CPOR), annual savings, and payback horizon.
  4. Computes empirical **95% Confidence Intervals (2.5th to 97.5th percentiles)**:
     * Baseline CPOR: ₹128.80 (95% CI: [112.41, 160.12]).
     * AI-Optimized CPOR: ₹124.30 (95% CI: [104.46, 159.92]).
     * Annual Net Savings: ₹157,450 / year.
     * Capital Payback Horizon: 16.4 months (median).
  5. Runs sensitivity regression:  
     $$\text{CPOR Savings} = 1.83 + (3.67 \times \text{Occupancy}) + (0.0001 \times \text{Wage})$$
  6. Exports [Table 3](file:///home/pranav/HotelManagement_Project/results/paper1/tables/table3_techno_economic_monte_carlo_india.csv).

---

#### Component 6: NLP Sentiment & Personalization (`nlp_sentiment.py`)
* **Path**: [`src/paper2_sustainability/nlp_sentiment.py`](file:///home/pranav/HotelManagement_Project/src/paper2_sustainability/nlp_sentiment.py)
* **What it does**:
  1. Benchmarks rule-based lexicon sentiment (VADER) against supervised TF-IDF (10,000 n-gram features) + LinearSVC:
     * **TF-IDF + LinearSVC**: Accuracy $= 85.68\%$, Macro-F1 $= 0.6844$.
     * **VADER Baseline**: Accuracy $= 77.24\%$, Macro-F1 $= 0.4700$.
  2. Extracts granular aspect polarity across: Cleanliness, Bathroom, Linen, Odor, and Promptness.
  3. Runs $K$-Means clustering ($K=4$, Silhouette Score $= 0.418$) to derive four distinct guest personas:
     * *Eco-Conscious Minimalist* (17.1%)
     * *Business Short-Stay* (34.0%)
     * *Family High-Intensity* (26.6%)
     * *Luxury High-Touch* (22.2%)
  4. Establishes a dynamic service rule matrix mapping each cluster to customized cleaning frequencies and linen policies.
  5. Exports [Table 4](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table4_nlp_sentiment_benchmark.csv), [Table 5](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table5_guest_segment_profiles.csv), and [Table 6](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table6_personalization_rule_matrix.csv).

---

#### Component 7: Calibrated Environmental ML (`environmental_models.py`)
* **Path**: [`src/paper2_sustainability/environmental_models.py`](file:///home/pranav/HotelManagement_Project/src/paper2_sustainability/environmental_models.py)
* **What it does**:
  1. Trains multivariate regressors (Random Forest vs. Gradient Boosting) on 250 lodging properties to predict Site EUI (kBtu/sqft).
  2. Evaluates two eco-housekeeping interventions:
     * **Smart HVAC Setback**: Thermostat setback by 2°C during cleaning saves 1.65 kWh per serviced room.
     * **Personalized Eco-Stayover Opt-Out**: 35% opt-out among stayover guests saves 72 L of laundry water and 2.8 kWh of water-heating power per room-night.
  3. Quantifies annual savings for a 120-room property:
     * **Electricity Saved**: 70,578 kWh/year (12.10% reduction).
     * **Water Saved**: 476,824 Liters/year (4.32% reduction).
     * **Avoided Carbon (India CEA)**: **51.40 Metric Tonnes $\text{CO}_2\text{e}$**.
     * **Avoided Carbon (US EPA)**: **28.04 Metric Tonnes $\text{CO}_2\text{e}$**.
  4. Exports [Table 7](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table7_energy_regression_benchmark.csv) and [Table 8](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table8_carbon_resource_abatement_ledger.csv).

---

#### Component 8: TAM Survey & Barrier Analysis (`tam_survey_analysis.py`)
* **Path**: [`src/paper2_sustainability/tam_survey_analysis.py`](file:///home/pranav/HotelManagement_Project/src/paper2_sustainability/tam_survey_analysis.py)
* **What it does**:
  1. Analyzes survey data from 32 hospitality leadership executives (Executive Housekeepers, General Managers, IT Leads).
  2. Computes Cronbach's alpha internal consistency reliability across five constructs: Perceived Usefulness, Ease of Use, Cost Constraints, Tech Resistance, and Privacy.
  3. Computes the **Relative Importance Index (RII)** to rank adoption roadblocks:
     * Rank 1: *Digital Literacy Gap among Housekeeping Crew* (Mean 4.31, RII = 0.862, Tier 1).
     * Rank 2: *High Upfront IoT Hardware / Retrofit Cost* (Mean 4.22, RII = 0.844, Tier 1).
     * Rank 3: *Unclear Short-Term ROI Timeline* (Mean 3.84, RII = 0.769, Tier 2).
     * Rank 4: *Staff Surveillance Anxiety* (Mean 3.81, RII = 0.762, Tier 2).
     * Rank 5: *Guest Privacy Concerns* (Mean 3.62, RII = 0.725, Tier 2).
  4. Formulates a qualitative managerial mitigation taxonomy (e.g., Hardware-as-a-Service leasing, optical-blind PIR sensors).
  5. Exports [Table 9](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table9_tam_psychometric_reliability.csv), [Table 10](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table10_adoption_barrier_hierarchy.csv), and [Table 11](file:///home/pranav/HotelManagement_Project/results/paper2/tables/table11_managerial_mitigation_framework.csv).

---

#### Component 9: Publication Figures (`generate_figures.py`)
* **Path**: [`src/visualization/generate_figures.py`](file:///home/pranav/HotelManagement_Project/src/visualization/generate_figures.py)
* **What it does**:
  * Programmatically renders six multi-panel figures at **300 DPI** matching IEEE and Elsevier publishing specifications (saved in both high-res PNG and vector PDF format):
    * **Figure 1**: Forecasting Benchmark Trajectory & Parity Plot.
    * **Figure 2**: Floor Makespan & Overtime Distributions (Boxplot & Barplot).
    * **Figure 3**: Monte Carlo Stochastic CPOR Density & Payback Histogram.
    * **Figure 4**: Housekeeping 5-Aspect Sentiment Radar Plot.
    * **Figure 5**: Annual Resource & Carbon Abatement Ledger (Dual-Panel).
    * **Figure 6**: Empirical Barrier Hierarchy (Horizontal Bar Chart).

---

#### Component 10: Streamlit Research Dashboard (`app.py`)
* **Path**: [`src/dashboard/app.py`](file:///home/pranav/HotelManagement_Project/src/dashboard/app.py)
* **What it does**:
  * Multi-page web dashboard providing interactive decision support across both papers:
    * Executive overview and architecture diagram.
    * Real-time sliders for hotel rooms, occupancy, housekeeper wage, and IoT CapEx with instant payback recalculation.
    * Aspect sentiment radar viewer and guest segment explorer.
    * **Publication Center**: Dropdown allowing users to preview any of the 11 tables and copy formatted LaTeX code directly into paper manuscripts.

---

#### Component 11: Master Orchestration (`run_pipeline.py`)
* **Path**: [`run_pipeline.py`](file:///home/pranav/HotelManagement_Project/run_pipeline.py)
* **What it does**:
  * Connects all data processing, ML model training, constraint solving, Monte Carlo simulation, and figure rendering scripts into an automated pipeline.
  * Runs sequentially with execution time logging:
    ```bash
    .venv/bin/python run_pipeline.py
    ```

---

### 5. How to Run, Test, and Reproduce

#### Environment Setup
The project uses an isolated virtual environment created with `uv` and Python 3.11:
```bash
# Activate the environment
source .venv/bin/activate
```

#### Re-Running the Experiment Pipeline
To execute all models, simulations, and figure generators:
```bash
python run_pipeline.py
```
*Expected Runtime*: ~90 seconds. All outputs will be saved to `results/paper1/` and `results/paper2/`.

#### Launching the Interactive Dashboard
To start the Streamlit research interface:
```bash
streamlit run src/dashboard/app.py
```
Open `http://localhost:8501` in your browser.

#### Running Individual Modules Standalone
You can run any module independently:
```bash
# Run workload forecasting benchmark only
python src/paper1_operations/forecasting.py

# Run OR-Tools scheduling simulation only
python src/paper1_operations/scheduler.py

# Run Monte Carlo cost simulation only
python src/paper1_operations/cost_monte_carlo.py

# Run NLP sentiment mining & guest clustering only
python src/paper2_sustainability/nlp_sentiment.py

# Run environmental regressions only
python src/paper2_sustainability/environmental_models.py

# Run TAM survey psychometrics only
python src/paper2_sustainability/tam_survey_analysis.py

# Regenerate all 300 DPI figures
python src/visualization/generate_figures.py
```

---

### 6. Developer Extension Guide

#### How to Add a New Forecasting Model
1. Open [`src/paper1_operations/forecasting.py`](file:///home/pranav/HotelManagement_Project/src/paper1_operations/forecasting.py).
2. Inside `train_and_benchmark_forecasting()`, initialize your model (e.g., `CatBoostRegressor`):
   ```python
   from catboost import CatBoostRegressor
   cb = CatBoostRegressor(iterations=200, random_seed=42, verbose=0)
   cb.fit(X_train, y_train)
   pred_cb = cb.predict(X_test)
   results['CatBoost'] = {
       'MAE': round(mean_absolute_error(y_test, pred_cb), 2),
       'RMSE': round(np.sqrt(mean_squared_error(y_test, pred_cb)), 2),
       'MAPE_%': round(np.mean(np.abs((y_test - pred_cb) / np.maximum(y_test, 1))) * 100, 2),
       'R2': round(r2_score(y_test, pred_cb), 4)
   }
   ```
3. Re-run `python src/paper1_operations/forecasting.py`. Table 1 and Figure 1 will automatically update.

#### How to Add a New Scheduling Constraint
1. Open [`src/paper1_operations/scheduler.py`](file:///home/pranav/HotelManagement_Project/src/paper1_operations/scheduler.py).
2. Inside `run_ortools_cpsat_scheduler()`, add your constraint to `model`:
   * *Example*: Housekeeper skill certification for luxury suites:
     ```python
     # If staff s is not certified for suites, enforce x[(t, s)] == 0 for suite tasks
     for t in range(num_tasks):
         if tasks[t]['is_suite']:
             for s in uncertified_staff:
                 model.Add(x[(t, s)] == 0)
     ```
3. The solver will automatically re-optimize subject to the new constraint.
