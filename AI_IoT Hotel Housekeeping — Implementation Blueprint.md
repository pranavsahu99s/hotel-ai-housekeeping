# AI/IoT hotel housekeeping — implementation blueprint

A practical build plan: architecture, datasets, models, tech stack, and pitfalls — mapped to your five research objectives.

[1. System architecture](#arch) [2. Dataset plan — exact sources per objective](#data) [3. Models & algorithms per objective](#models) [4. Data schema](#schema) [5. Technical stack](#stack) [6. Dashboard design](#dashboard) [7. Evaluation metrics for the paper](#metrics) [8. Phased implementation roadmap](#roadmap) [9. Things to remember / tackle head-on](#pitfalls)

## 1System architecture

Treat this as **four linked sub-systems feeding one evaluation layer** — not one giant model. That is the single decision that makes the rest of this plan buildable in a semester.

**Why this shape, not one pipeline:** objectives 1, 3, 4 share a common backbone (occupancy/booking data → simulation → optimization → cost/energy deltas), objective 2 is a separate NLP track on review text, and objective 5 is qualitative survey research that never touches your ML code. Wiring all five into one "AI engine" is the most common way these projects stall — keep them as independent, swappable modules that report into a shared evaluation layer.

## 2Dataset plan — exact sources per objective

There is no single public dataset that covers "hotel housekeeping operations" end to end. Real data covers guest sentiment and building energy well; it does not cover live IoT sensor streams or per-room cleaning times. Below is the honest map — what's real, what you'll have to simulate, and exactly where to get each.

real data you generate/simulate — carry these tags into your methodology section. Reviewers will ask; label it up front instead.

### Objective 1 & 3 — Operational efficiency and cost-effectiveness

| Dataset | Source | What it gives you | Status |
| --- | --- | --- | --- |
| **Hotel Booking Demand** (119,390 bookings, city + resort hotel, Portugal, 2015–17) | Kaggle: `jessemostipak/hotel-booking-demand` | Check-in/out dates, lead time, stay length, room type, cancellations, special requests — everything you need to derive daily room-turnover volume and workload patterns | real |
| **OYO Hotel Rooms** (India, \~460 properties) | Kaggle: `sonu1maheshwari/oyo-hotel-rooms` | Pricing, discounts, ratings for budget Indian hotel chain — use for an India-specific cost baseline | real |
| **Hotels in Munnar, Kerala** (MakeMyTrip scrape) | Kaggle: `andrewgeorgeissac/hotels-in-munnar-kerala` | India-specific pricing + review counts for a second regional comparison point | real |
| **Per-room cleaning time labels** (checkout clean vs. stay-over clean, by room type) | You generate this | No public dataset has this. Simulate durations calibrated to published industry standards: ISSA (International Sanitary Supply Association) cleaning-time benchmarks and AHLA/HFTP housekeeping productivity standards (typically 20–30 min stay-over, 30–45 min checkout clean per room — cite the specific standard you use) | simulated |
| **Staff/shift data** (headcount, wages, shift patterns) | You generate this | Parameterize from published Indian hospitality wage data (e.g. FHRAI/NCS wage surveys) or US BLS housekeeping wage data if using an international comparison | simulated |

### Objective 2 — Guest satisfaction & personalization

| Dataset | Source | What it gives you | Status |
| --- | --- | --- | --- |
| **515K Hotel Reviews Data in Europe** (1,493 hotels, Booking.com) | Kaggle: `jiashenliu/515k-hotel-reviews-data-in-europe` | Positive/negative review text, reviewer nationality, score, tags — your strongest real-data pillar. Filter for housekeeping-relevant keywords (clean, towel, smell, dust, tidy, room service) for aspect-based sentiment | real |
| **Air India / hospitality TripAdvisor-style review sets** and equivalent hotel review corpora on Kaggle (search "hotel reviews India", "TripAdvisor hotel reviews") | Kaggle (multiple, verify size/date before choosing) | India-specific review sentiment to compare against the Europe set — this is what lets you make an India-vs-foreign claim honestly | real |
| **MakeMyTrip / Goibibo review scrape** (if the above proves too thin) | Apify actors (`factden/makemytrip-scraper`, `shahidirfan/makemytrip-hotel-reviews-scraper`) — paid, \~$1.50–2.50/1,000 reviews | Fresh India-specific reviews with sub-ratings (cleanliness, facilities, value) already broken out — directly usable, no aspect extraction needed for the cleanliness sub-score | real (paid API) |

**Scraping caution:** Booking.com/TripAdvisor ToS generally prohibit direct scraping. Use the pre-packaged Kaggle datasets (already public, licensed for reuse) as your primary source, and only use a paid scraper API (Apify) if you need fresher India-specific data — that's a sanctioned, ToS-compliant route since the platform itself sells the access.

### Objective 4 — Environmental impact

| Dataset | Source | What it gives you | Status |
| --- | --- | --- | --- |
| **ASHRAE Great Energy Predictor III (GEPIII) / Building Data Genome 2** — 1,448 buildings, 16 sites, includes a lodging/residential-hall category | Kaggle competition: `ashrae-energy-prediction`; raw source: Building Data Genome 2 project (GitHub, `buds-lab/ashrae-great-energy-predictor-3-solution-analysis`) | Real hourly electricity/steam/chilled-water/hot-water meter data + weather, filterable to lodging-type buildings — your calibration source for energy-per-occupied-room baselines | real |
| **NYC Local Law 84 Energy & Water Data Disclosure** (annual, includes Property Type = Hotel with Site EUI and water use intensity) | NYC Open Data portal | Real, hotel-specific building-level energy and water benchmarks — the single best "real hotel environmental data" source publicly available | real |
| **India Bureau of Energy Efficiency (BEE) hotel star-rating scheme data / published case studies** | BEE (beeindia.gov.in) reports, UNWTO Hotel Energy Solution toolkit | Indian-context energy benchmarks (kWh/room-night by hotel category) to contextualize your India findings instead of only using Western benchmarks | real (published aggregates, not raw meter data) |
| Water/linen/chemical consumption per cleaning type | You generate this | Calibrate from Cornell Center for Hospitality Research sustainability studies and published green-hotel case studies (cite specific figures, e.g. liters/load for laundry, ml/room for chemicals) | simulated |

### Objective 5 — Barriers & challenges

| Source | What it gives you | Status |
| --- | --- | --- |
| Structured questionnaire (Likert + open text) to 15–30 hotel managers/housekeeping heads, India-focused, distributed via LinkedIn hospitality groups, FHRAI contacts, or your institution's industry network | Primary data — barrier frequency/importance ranking, thematic quotes | real (primary, self-collected) |
| Existing published surveys on hotel technology adoption barriers (search Google Scholar: "AI adoption barriers hospitality India", "IoT hotel technology adoption survey") | Triangulation/secondary support if your own sample is small — cite as prior evidence, don't merge raw numbers with your own survey | real (secondary) |

Verify current dataset URLs on Kaggle before download — usernames/slugs occasionally change, and always check the dataset's license tab (most of the above are CC0 or ODbL) before use in a publication.

## 3Models & algorithms per objective

### Objective 1 Operational efficiency

#### Demand / workload forecasting

- **Baseline:** SARIMA or Facebook Prophet on daily check-out counts (captures weekly/seasonal booking patterns) — simple, interpretable, good for the paper's "classical benchmark" comparison.
- **Main model:** Gradient-boosted trees — **XGBoost or LightGBM regressor** — predicting daily room-turnover volume and per-room cleaning-priority score from booking features (lead time, stay length, room type, day-of-week, season, is_repeated_guest). Tabular + gradient boosting beats deep learning on datasets this size (119K rows) — don't reach for an LSTM here, it will overfit and won't outperform XGBoost on tabular data.

#### Staff–task scheduling optimization

- **Core method:** **Constraint programming via Google OR-Tools CP-SAT** — assign predicted cleaning tasks to available staff minimizing total overtime + idle time, subject to shift-length and skill constraints. This is the right tool for the job (discrete assignment with hard constraints), not a neural network.
- **Baseline to beat it:** a simple rule-based/greedy scheduler (assign tasks in check-out order, round-robin staff) — you need this baseline explicitly coded, it's what "AI-optimized vs. baseline" means in your results section.
- **Optional stretch (only if time allows):** a reinforcement-learning agent (PPO via `stable-baselines3`) that learns dynamic reassignment as cancellations/early check-ins occur — genuinely more novel/publishable, but treat as a secondary experiment, not the core deliverable.

#### Simulation backbone

- **SimPy** discrete-event simulation of a hotel floor: rooms, staff, cleaning tasks, arrival/departure events drawn from the booking dataset's empirical distributions. This is what lets you run "baseline schedule" and "AI-optimized schedule" as two simulation runs and compare KPIs — the backbone of your objective 1 and 3 results.

### Objective 2 Guest satisfaction & personalization

#### Sentiment / aspect extraction

- **Baseline:** VADER or TextBlob rule-based sentiment on review text — fast, no training needed, good sanity check.
- **Main model:** Fine-tune **DistilBERT or RoBERTa-base** (HuggingFace `transformers`) on the review dataset for sentiment classification; combine with keyword/aspect filtering (clean, towel, smell, tidy, housekeeping, service) to isolate housekeeping-specific sentiment from overall review sentiment — this is the step that makes the analysis about *housekeeping* specifically, not hotels in general.
- **Topic discovery:** **BERTopic** (embeddings + clustering, better than classic LDA on short review text) to surface recurring housekeeping complaint/praise themes automatically — gives you a nice figure for the paper (topic clusters with example terms).

#### Personalization layer

- **Guest segmentation:** **K-Means clustering** on engineered guest-preference features (derived from review aspect scores + booking metadata: stay length, room type, repeat-guest flag) into 3–5 segments (e.g., minimal-service/eco-conscious, high-touch/luxury-expectation, business-short-stay, family/allergy-sensitive).
- **Recommendation logic:** a simple **content-based rule engine** mapping each segment to a housekeeping service profile (cleaning frequency, product type, do-not-disturb defaults) — you do not need a deep recommender system here; a transparent rule table is more defensible for a paper claiming "personalization" than a black-box model with no ground truth to validate against.

### Objective 3 Cost-effectiveness

- **Core method:** a **cost simulation model** layered on top of the objective-1 SimPy output — labor cost = (staff-hours from baseline vs. AI-optimized schedule) × wage rate; supply cost = (cleaning events × per-clean product usage) × unit cost. Run as a **Monte Carlo simulation** (vary occupancy, wage, and supply-price assumptions across distributions) so you can report cost savings with a confidence interval, not a single point estimate — this is what makes it defensible in a paper.
- **Supporting regression:** a small **linear/ridge regression** predicting cost-per-occupied-room from occupancy rate and staffing ratio, fit on your simulated scenarios — useful for a clean "sensitivity" chart.
- **ROI/payback:** combine simulated savings with a stated technology investment figure (cite BEE/industry reports for typical IoT-retrofit costs in Indian hotels) to compute a payback period — this single number is often the most quoted result in this type of paper.

### Objective 4 Environmental impact

- **Energy/water regression:** **Random Forest or XGBoost regressor** trained on the ASHRAE/BDG2 lodging subset to predict Site EUI (energy use intensity) from occupancy rate, building size, and climate zone — then apply it to your simulated baseline vs. AI-optimized occupancy/cleaning-frequency scenarios to estimate the consumption delta.
- **Calibration, not just prediction:** anchor your final numbers to the NYC LL84 hotel benchmarks and BEE India figures so your reported "% reduction" is expressed against a real published baseline (e.g., "X kWh/occupied room-night vs. the NYC hotel median of Y") — this is what turns a model output into a citable result.
- **CO2e conversion:** apply standard grid emission factors (India CEA baseline database, or EPA eGRID for a US comparison) to convert energy savings into a carbon-reduction figure — reviewers like this because it translates your ML result into a policy-relevant unit.

### Objective 5 Barriers & challenges

- **This is not a modeling objective — treat it as applied qualitative research.** Design the questionnaire around an established technology-adoption framework (**TAM** — Technology Acceptance Model, or **UTAUT**) so your barrier categories map to a citable theoretical structure instead of being ad hoc.
- **Analysis:** descriptive statistics + ranking on the Likert items (report Cronbach's alpha for internal consistency if you have 15+ respondents), and **thematic coding** (manual, or assisted with a zero-shot classification pass using an LLM to pre-cluster open-text responses into themes you then verify by hand — don't publish LLM-only coding as your qualitative analysis).
- **If your primary sample is small (a real risk):** explicitly frame it as an exploratory/pilot study and triangulate with 2-3 existing published surveys on hospitality tech adoption in India — this is normal practice and reviewers respect it far more than an inflated claim on n=12.

## 4Data schema

A minimal relational schema that ties all five objectives to the same hotel/room/booking keys — this is what lets your dashboard join across tabs instead of five disconnected CSVs.

```
hotels(id, name, country, city, star_rating, property_type)
rooms(id, hotel_id, room_type, floor)
bookings(id, hotel_id, room_id, checkin_date, checkout_date, lead_time,
         adults, children, is_repeated_guest, is_canceled, market_segment)
cleaning_events(id, room_id, booking_id, event_date, clean_type
                [stayover|checkout|deep], predicted_duration_min,
                simulated_actual_duration_min, staff_id, schedule_type
                [baseline|ai_optimized])
staff(id, hotel_id, shift, hourly_wage_inr, role)
reviews(id, hotel_id, review_text, rating, review_date,
        housekeeping_sentiment_score, aspect_tags[])
guest_segments(booking_id, segment_label, segment_confidence)
energy_records(id, hotel_id_or_proxy_building_id, date, meter_type,
                consumption_kwh, site_eui, water_liters)
survey_responses(id, respondent_role, region, barrier_ratings_json,
                  open_text, theme_tags[])
```

Use PostgreSQL for the structured tables above; store raw review/survey text separately (a JSON/text column is fine at this scale — you don't need MongoDB unless your review corpus exceeds a few hundred thousand rows).

## 5Technical stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | Python 3.10+ | Every library below is Python-first; no reason to split languages for a project this scale |
| Data wrangling | pandas, numpy | Standard; your datasets (max \~500K rows) don't need Spark/Dask |
| Classical ML | scikit-learn, XGBoost, LightGBM | Gradient boosting is the right tool for every tabular prediction task here |
| Time series | statsmodels (SARIMA), Prophet | Demand forecasting baselines |
| NLP | HuggingFace transformers (DistilBERT/RoBERTa), BERTopic, spaCy, VADER | Sentiment + topic modeling on review text |
| Optimization | Google OR-Tools (CP-SAT), PuLP | Staff-task scheduling as constraint satisfaction, not ML |
| Simulation | SimPy | Discrete-event simulation of hotel floor operations — the backbone connecting objectives 1, 3, 4 |
| RL (optional/stretch) | stable-baselines3 (PPO) | Only if you have time after the core deliverables work |
| Backend/API | FastAPI + Uvicorn | Serves precomputed model outputs to the dashboard as JSON; lightweight, async, auto-generates docs |
| Database | PostgreSQL (or SQLite for a solo/coursework build) | Structured relational schema above |
| Dashboard | **Streamlit** (fastest to build) or Plotly Dash; React + Recharts if you want a portfolio-polished custom UI | Streamlit gets you from "model output" to "interactive dashboard" in hours, not weeks — start here, upgrade to React only if you need it to look production-grade |
| Experiment tracking | MLflow (optional but recommended) | Logs model versions/metrics — makes your "methodology" section in the paper reproducible and gives you clean comparison tables for free |
| Reproducibility | Git + a `requirements.txt`/`environment.yml`, fixed random seeds | Non-negotiable for a paper — a reviewer or committee member should be able to rerun your pipeline |
| Deployment (optional) | Docker; Streamlit Community Cloud or Render for a free public demo link | Useful if you want to share the dashboard beyond your own machine |

## 6Dashboard design

One tab per objective, all reading from the FastAPI serving layer (precomputed — never train live in the dashboard):

- **Tab 1 — Operational efficiency:** baseline vs. AI-optimized schedule side by side (Gantt-style staff/room chart), room-turnaround time distribution, staff utilization %.
- **Tab 2 — Guest satisfaction:** sentiment trend over time, housekeeping-aspect breakdown, guest segment distribution, sample flagged reviews per topic cluster.
- **Tab 3 — Cost-effectiveness:** cost-per-occupied-room comparison, Monte Carlo savings distribution (with confidence band), payback-period calculator with adjustable investment-cost slider.
- **Tab 4 — Environmental impact:** kWh and liters per occupied room-night, baseline vs. AI-optimized, benchmarked against NYC LL84 / BEE India reference lines, estimated CO2e avoided.
- **Tab 5 — Barriers:** ranked barrier chart from survey data, theme frequency, respondent breakdown by role/region — static/descriptive, not model-driven, and should say so on the tab.

Label every number in the UI with its provenance tag (real / simulated / estimated) — small footnote text under each chart. This single habit does more for your paper's credibility than any model choice.

## 7Evaluation metrics for the paper

| Objective | Metrics to report |
| --- | --- |
| 1. Operational efficiency | Avg. room turnaround time (baseline vs. AI), staff utilization %, task completion rate, overtime hours saved, forecast accuracy (MAE/RMSE for workload prediction) |
| 2. Guest satisfaction | Sentiment classification F1/accuracy, topic coherence score, cluster silhouette score, simulated satisfaction uplift from personalization rules |
| 3. Cost-effectiveness | Cost per occupied room (CPOR), % cost reduction with 95% CI (from Monte Carlo), payback period, ROI |
| 4. Environmental impact | kWh / occupied room-night, liters water / occupied room-night, % reduction vs. baseline, estimated CO2e avoided, regression R²/RMSE for the energy model |
| 5. Barriers | Barrier frequency ranking, Cronbach's alpha (questionnaire reliability), thematic frequency counts, sample size and response rate (report honestly) |

## 8Phased implementation roadmap

Week 1

**Scoping & literature.** Lock target venue/audience, read 3-5 comparable papers, finalize which datasets you're actually using (download and eyeball all of them this week — don't discover a dataset is unusable in week 6).

Weeks 2-3

**Data pipeline.** Clean and load booking, review, and energy datasets into the schema above. Stand up the survey instrument and start distribution now — response collection is slow, start it in parallel with everything else.

Weeks 3-4

**Simulation backbone.** Build the SimPy hotel-floor simulation calibrated to the booking dataset's empirical arrival/departure distributions and your documented cleaning-time assumptions.

Weeks 4-5

**Objective 1 models.** Demand forecaster (XGBoost) + OR-Tools scheduler, baseline vs. AI-optimized comparison run.

Weeks 5-6

**Objective 2 models.** Fine-tune sentiment model, run BERTopic, build guest segmentation + personalization rule table.

Weeks 6-7

**Objectives 3 & 4.** Cost Monte Carlo simulation, energy/water regression calibrated to ASHRAE + NYC LL84 + BEE benchmarks.

Weeks 7-8

**Dashboard integration.** FastAPI serving layer + Streamlit tabs, wired to real computed outputs (not mocked data).

Weeks 8-9

**Evaluation & write-up.** Finalize all metrics tables, close out survey analysis, draft methodology/results sections while the pipeline is fresh in your head.

Ongoing

**Objective 5 survey collection** runs the entire time in the background — it has the longest lead time of anything in this plan (recruiting respondents), so it must start in week 2, not week 8.

## 9Things to remember / tackle head-on

**Label real vs. simulated everywhere.** Every chart, every table, every number in the paper's results section should be traceable to "real dataset X" or "simulated, calibrated to source Y." This is the single biggest credibility lever you control.

**Temporal splits, not random splits.** For any forecasting model on the booking dataset, split train/test by date (e.g., train on 2015-16, test on 2017), not randomly — a random split leaks future information and will make your accuracy look better than it is.

**Don't merge cross-region data into one number.** Europe review sentiment, India pricing data, and US/global energy benchmarks come from different populations. Report objective-2 results by region, and don't claim a single "AI improved satisfaction by X%" figure that silently mixes Portuguese hotel reviews with Indian pricing data — a reviewer will catch this immediately.

**Document your synthetic-label generation process in detail.** Cleaning-time and cost figures you simulate need a clearly cited calibration source (ISSA standards, Cornell studies, BEE reports) and a stated methodology — treat this as its own subsection in your methods chapter, with a sensitivity analysis showing how results change if the assumption shifts ±20%.

**Always compare against an explicit baseline.** "Cost-effectiveness" and "efficiency" are meaningless without a baseline to be effective/efficient *relative to*. Code the naive/rule-based scheduler before the optimized one — it's a smaller task than it sounds and it's what makes every "% improvement" claim valid.

**Watch dataset licensing.** Most Kaggle datasets used here are CC0; scraper-based data (Apify) is paid and subject to the source platform's terms — check before publishing anything derived from it, and cite the original data source paper where one exists (e.g., the hotel booking demand dataset has a citable Data in Brief paper — cite that, not just "Kaggle").

**Treat objective 5's sample size honestly.** A 15-30 respondent survey is a pilot study, not a statistically powered study — say so explicitly rather than over-claiming significance, and lean on thematic/qualitative framing over inferential statistics.

**Fix random seeds and log package versions from day one.** Reproducibility questions are common in defenses/reviews — a `requirements.txt` and seeded runs cost you nothing now and save a scramble later.

**Resist scope creep.** Federated learning across hotel chains, live IoT hardware, or a full RL scheduling system are genuinely interesting — write them up as "future work," don't let them become blockers for the core five-objective deliverable.