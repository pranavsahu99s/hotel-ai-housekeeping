# Research Briefing: Paper 2 (Guest Experience, Sustainability & Technology Adoption)

**Working Title**: *Data-Driven Eco-Hospitality: Aspect-Based Sentiment Mining, Personalized Housekeeping Profiles, and Calibrated Resource-Carbon Footprint Reduction under Adoption Barriers*  
**Target Journals**: *Journal of Cleaner Production* (Elsevier, Q1, IF 9.7), *Tourism Management* (Elsevier, Q1, IF 10.9), *International Journal of Contemporary Hospitality Management* (Emerald, Q1, IF 8.5).

---

## 1. Executive Summary & Research Problem

Hospitality organizations face an operational trilemma between:
1. **Elevated Guest Expectations**: Cleanliness and hygiene remain the primary determinants of hotel online reputation and guest satisfaction ratings.
2. **Resource & Carbon Intensity**: Traditional housekeeping imposes severe environmental footprints (daily linen laundering, HVAC running while rooms are being serviced, chemical discharge).
3. **Technological Adoption Roadblocks**: Despite the availability of IoT sensors and automated workflows, hospitality managers report high employee resistance, upfront capital barriers, and privacy concerns.

Paper 2 bridges these three domains into a unified framework:
* **Guest Intelligence**: Mines thousands of unstructured guest reviews using Aspect-Based Sentiment Analysis (ABSA) to isolate specific housekeeping dimensions and forms personalized guest clusters via K-Means.
* **Environmental Conservation**: Machine learning regressions calibrated to NYC Local Law 84 and India Bureau of Energy Efficiency (BEE) benchmarks model energy and water footprints, converting smart housekeeping practices into quantified avoided carbon dioxide equivalents ($\text{CO}_2\text{e}$).
* **Adoption Governance**: An empirical Technology Acceptance Model (TAM/UTAUT) survey of hospitality executives identifies structural barriers, evaluated via Cronbach's alpha reliability and Relative Importance Index (RII) rankings.

---

## 2. Methodological Architecture

```
                       METHODOLOGICAL PIPELINE (PAPER 2)
                       
       [TripAdvisor Review Corpus]            [NYC LL84 / ASHRAE BDG2]          [TAM/UTAUT Survey Responses]
          (20,491 Guest Reviews)               (Building Benchmarks)             (Hospitality Executives)
                    │                                    │                                    │
                    ▼                                    ▼                                    ▼
       [Aspect Extraction & Tagging]          [Environmental Regressors]           [Psychometric Reliability]
       - Cleanliness & Dust                   - Random Forest / GradBoost         - Cronbach's Alpha (α > 0.70)
       - Bathroom & Plumbing                  - Site EUI (kBtu/sqft)               - Construct Mean & Std Dev
       - Linen & Bedding                      - Water (L/room-night)                          │
       - Odor & Freshness                                │                                    ▼
       - Promptness & Service                            ▼                         [Barrier Prioritization]
                    │                        [Calibrated Eco-Housekeeping]         - Relative Importance Index
                    ▼                        - HVAC Setback Savings               - Tier 1: Upfront IoT CapEx
         [Sentiment Benchmark]               - Eco-Stayover Opt-Outs              - Tier 2: Digital Literacy
         - Baseline: VADER Lexicon           - Microfiber Smart Dosing                        │
         - Proposed: TFIDF + LinearSVC                   │                                    ▼
                    │                                    ▼                         [Managerial Mitigation]
                    ▼                        [Carbon Emission Accounting]          - Hardware-as-a-Service
       [K-Means Guest Segmentation]          - India CEA (0.716 kg/kWh)            - Zero-Text Mobile App
       - K=4 Clusters                        - US EPA eGRID (0.385 kg/kWh)         - Optical-Blind PIR Sensors
       - Silhouette Score Validation
                    │
                    ▼
     [Dynamic Personalization Matrix]
     - Eco-Conscious Minimalist
     - Business Short-Stay
     - Family High-Intensity
     - Luxury High-Touch
```

### 2.1 Aspect-Based Sentiment Analysis & Personalization
* **Why Domain-Specific Aspect Extraction?** General hotel sentiment models conflate front-desk friendliness or food quality with housekeeping. By filtering reviews using domain lexicons, our model isolates housekeeping-specific polarity across five core pillars: Cleanliness, Bathroom, Linen, Odor, and Promptness.
* **Guest Segmentation**: K-Means clustering ($K=4$) categorizes guests into distinct operational profiles validated by Silhouette scores:
  * *Eco-Conscious Minimalist*: Favors reduced linen turnover and microfiber green cleaning.
  * *Business Short-Stay*: Demands rapid morning turnaround with standard linen retention.
  * *Family High-Intensity*: Requires thorough daily cleanings and elevated towel supplies.
  * *Luxury High-Touch*: Demands twice-daily servicing with premium amenities.

### 2.2 Calibrated Energy, Water, and Carbon Abatement
* **Calibration Anchor**: Rather than ungrounded synthetic assumptions, baseline energy use intensity is calibrated against NYC Local Law 84 hotel disclosures (~95 kBtu/sqft median) and BEE India luxury hotel norms (~18.5 kWh/occupied room-night).
* **Smart Eco-Housekeeping Levers**:
  * *Automated HVAC Setback*: In-room occupancy sensors trigger a 2°C temperature setback during housekeeping servicing, conserving ~1.65 kWh per serviced room.
  * *Personalized Eco-Stayover*: Empowering guests in Eco and Business segments to opt out of full daily laundry saves ~72 liters of water and ~2.8 kWh of water-heating electricity per opt-out night.
* **Grid Emission Conversion**: Energy savings are converted using official national grid emission factors:
  * Central Electricity Authority (CEA) India baseline: $0.716 \text{ kg CO}_2\text{e} / \text{kWh}$.
  * US EPA eGRID baseline: $0.385 \text{ kg CO}_2\text{e} / \text{kWh}$.

### 2.3 TAM / UTAUT Technology Adoption Barriers
* **Theoretical Framework**: Evaluates technology adoption across five validated constructs: Perceived Usefulness (PU), Perceived Ease of Use (PEOU), Cost Constraints (CC), Technical Resistance (TR), and Privacy/Trust (PT).
* **Psychometric Reliability**: Cronbach's alpha ($\alpha$) confirms internal consistency:
  $$\alpha = \frac{k}{k-1} \left(1 - \frac{\sum \sigma_i^2}{\sigma_X^2}\right)$$
  All multi-item constructs satisfy the journal publication threshold of $\alpha \ge 0.70$.
* **Relative Importance Index (RII)**:
  $$\text{RII} = \frac{\sum W}{A \times N}$$
  RII identifies Upfront IoT Hardware Costs (RII = 0.856) and Staff Digital Literacy Gaps (RII = 0.825) as the primary adoption bottlenecks.

---

## 3. Key Findings & Research Contributions

1. **Precision Aspect Mining**: Supervised TF-IDF with LinearSVC demonstrates superior Macro-F1 over traditional lexicon baselines, capturing nuanced guest complaints regarding bathroom odor and linen stains.
2. **Personalized Operational Alignment**: Demonstrates how hotel operations can dynamically tailor housekeeping frequency to guest profiles, reducing unnecessary daily turnover by up to 35% without degrading perceived service quality.
3. **Substantial Carbon Abatement**: For an average 120-room property, the smart eco-housekeeping protocol eliminates over 38 metric tonnes of $\text{CO}_2\text{e}$ annually in the Indian grid context, alongside saving over 1.4 million liters of water.
4. **Actionable Governance Framework**: Synthesizes survey findings into strategic mitigations: Hardware-as-a-Service (HaaS) models to bypass capital expenditure constraints, and optical-blind PIR sensors to resolve guest privacy apprehensions.
