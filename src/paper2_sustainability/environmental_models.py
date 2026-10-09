"""
Paper 2 Module: Calibrated Energy, Water, and Carbon Emission Abatement
Trains multivariate machine learning regressors on building energy data (NYC LL84 / ASHRAE BDG2).
Evaluates resource conservation under AI/IoT smart housekeeping vs. traditional baselines
and converts energy savings into CO2e emissions using CEA (India) and EPA eGRID factors.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CALIBRATION_DIR = os.path.join(BASE_DIR, "data", "calibration")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "paper2")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def load_calibration_standards():
    standards_path = os.path.join(CALIBRATION_DIR, "standards.json")
    with open(standards_path, 'r') as f:
        return json.load(f)

def benchmark_environmental_regressors():
    print("=== [PAPER 2] ENVIRONMENTAL ML REGRESSION BENCHMARK ===")
    data_path = os.path.join(PROCESSED_DIR, "hotel_energy_benchmarks.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Energy benchmarks not found at {data_path}. Run fetch_data.py first.")
        
    df = pd.read_csv(data_path)
    
    # Train/Test Split (80/20)
    features = ['star_rating', 'gross_floor_area_sqft', 'total_rooms', 'annual_occupancy_rate']
    target_eui = 'site_eui_kbtu_sqft'
    target_water = 'water_liters_per_room_night'
    
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split = int(0.8 * len(df))
    train_idx, test_idx = indices[:split], indices[split:]
    
    X_train, X_test = df.loc[train_idx, features], df.loc[test_idx, features]
    y_train_eui, y_test_eui = df.loc[train_idx, target_eui], df.loc[test_idx, target_eui]
    y_train_wat, y_test_wat = df.loc[train_idx, target_water], df.loc[test_idx, target_water]
    
    # 1. Baseline: Median Benchmark
    median_eui = y_train_eui.median()
    pred_med_eui = np.full_like(y_test_eui, median_eui)
    
    # 2. Random Forest Regressor
    rf_eui = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf_eui.fit(X_train, y_train_eui)
    pred_rf_eui = rf_eui.predict(X_test)
    
    # 3. Gradient Boosting Regressor
    gb_eui = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    gb_eui.fit(X_train, y_train_eui)
    pred_gb_eui = gb_eui.predict(X_test)
    
    eui_benchmark = {
        'Median_Benchmark': {
            'MAE': round(mean_absolute_error(y_test_eui, pred_med_eui), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_test_eui, pred_med_eui)), 2),
            'R2': round(r2_score(y_test_eui, pred_med_eui), 4)
        },
        'Random_Forest': {
            'MAE': round(mean_absolute_error(y_test_eui, pred_rf_eui), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_test_eui, pred_rf_eui)), 2),
            'R2': round(r2_score(y_test_eui, pred_rf_eui), 4)
        },
        'Gradient_Boosting': {
            'MAE': round(mean_absolute_error(y_test_eui, pred_gb_eui), 2),
            'RMSE': round(np.sqrt(mean_squared_error(y_test_eui, pred_gb_eui)), 2),
            'R2': round(r2_score(y_test_eui, pred_gb_eui), 4)
        }
    }
    
    df_eui_bench = pd.DataFrame(eui_benchmark).T
    print("\n--- SITE EUI (ENERGY INTENSITY) REGRESSION BENCHMARK ---")
    print(df_eui_bench.to_markdown())
    
    df_eui_bench.to_csv(os.path.join(TABLES_DIR, "table7_energy_regression_benchmark.csv"))
    with open(os.path.join(RESULTS_DIR, "energy_ml_metrics.json"), 'w') as f:
        json.dump(eui_benchmark, f, indent=2)
        
    return gb_eui

def calculate_carbon_and_resource_abatement(hotel_rooms: int = 120, annual_occupancy: float = 0.72):
    """
    Computes calibrated annual water, electricity, chemical, and CO2e abatement
    comparing traditional housekeeping operations against the AI/IoT eco-housekeeping protocol.
    """
    print(f"\n=== [PAPER 2] CALIBRATING ANNUAL CARBON & RESOURCE ABATEMENT (120-ROOM PROPERTY) ===")
    standards = load_calibration_standards()
    env = standards['environmental_standards']
    
    base_kwh_per_room = env['baselines']['energy_kwh_per_occupied_room_night']
    base_water_per_room = env['baselines']['water_liters_per_occupied_room_night']
    
    # Reductions
    hvac_setback_kwh = env['smart_reductions']['hvac_setback_during_cleaning_kwh']
    eco_stayover_kwh = env['smart_reductions']['iot_eco_stayover_energy_saved_kwh']
    towel_water_saved = env['smart_reductions']['towel_linen_reuse_water_saved_liters']
    
    # Emission Factors
    cea_factor = env['emission_factors']['india_grid_kg_co2_per_kwh']
    epa_factor = env['emission_factors']['us_grid_kg_co2_per_kwh']
    water_co2_factor = env['emission_factors']['water_treatment_kg_co2_per_kiloliter']
    
    # Annual occupied room nights
    annual_room_nights = hotel_rooms * 365 * annual_occupancy
    
    # 1. BASELINE Operational Resource Footprint
    baseline_annual_kwh = annual_room_nights * base_kwh_per_room
    baseline_annual_water_liters = annual_room_nights * base_water_per_room
    baseline_india_co2e_kg = (baseline_annual_kwh * cea_factor) + ((baseline_annual_water_liters / 1000.0) * water_co2_factor)
    baseline_us_co2e_kg = (baseline_annual_kwh * epa_factor) + ((baseline_annual_water_liters / 1000.0) * water_co2_factor)
    
    # 2. AI/IOT ECO-HOUSEKEEPING SAVINGS
    # Smart HVAC setbacks trigger during all room services (100% of rooms)
    annual_kwh_saved_hvac = annual_room_nights * hvac_setback_kwh
    
    # Guest Eco-Stayover opt-outs (35% participation rate across stayovers ~ 60% of total rooms)
    eco_stayover_nights = annual_room_nights * 0.60 * 0.35
    annual_kwh_saved_laundry = eco_stayover_nights * eco_stayover_kwh
    total_annual_kwh_saved = annual_kwh_saved_hvac + annual_kwh_saved_laundry
    
    # Water savings from towel/linen opt-outs
    total_annual_water_saved_liters = eco_stayover_nights * towel_water_saved
    
    # Avoided CO2e Emissions
    avoided_india_co2e_kg = (total_annual_kwh_saved * cea_factor) + ((total_annual_water_saved_liters / 1000.0) * water_co2_factor)
    avoided_us_co2e_kg = (total_annual_kwh_saved * epa_factor) + ((total_annual_water_saved_liters / 1000.0) * water_co2_factor)
    
    energy_reduction_pct = (total_annual_kwh_saved / baseline_annual_kwh) * 100
    water_reduction_pct = (total_annual_water_saved_liters / baseline_annual_water_liters) * 100
    
    abatement_ledger = [
        {
            'Environmental_Indicator': 'Electricity Consumption (kWh)',
            'Baseline_Annual': f"{baseline_annual_kwh:,.0f}",
            'AI_Optimized_Annual': f"{baseline_annual_kwh - total_annual_kwh_saved:,.0f}",
            'Net_Savings_Annual': f"{total_annual_kwh_saved:,.0f} kWh",
            'Reduction_%': f"{energy_reduction_pct:.2f}%"
        },
        {
            'Environmental_Indicator': 'Water Consumption (Liters)',
            'Baseline_Annual': f"{baseline_annual_water_liters:,.0f}",
            'AI_Optimized_Annual': f"{baseline_annual_water_liters - total_annual_water_saved_liters:,.0f}",
            'Net_Savings_Annual': f"{total_annual_water_saved_liters:,.0f} L",
            'Reduction_%': f"{water_reduction_pct:.2f}%"
        },
        {
            'Environmental_Indicator': 'India Grid Carbon Footprint (Metric Tonnes CO2e)',
            'Baseline_Annual': f"{baseline_india_co2e_kg / 1000.0:,.2f} MT",
            'AI_Optimized_Annual': f"{(baseline_india_co2e_kg - avoided_india_co2e_kg) / 1000.0:,.2f} MT",
            'Net_Savings_Annual': f"{avoided_india_co2e_kg / 1000.0:,.2f} MT",
            'Reduction_%': f"{(avoided_india_co2e_kg / baseline_india_co2e_kg) * 100:.2f}%"
        },
        {
            'Environmental_Indicator': 'US Grid Carbon Footprint (Metric Tonnes CO2e)',
            'Baseline_Annual': f"{baseline_us_co2e_kg / 1000.0:,.2f} MT",
            'AI_Optimized_Annual': f"{(baseline_us_co2e_kg - avoided_us_co2e_kg) / 1000.0:,.2f} MT",
            'Net_Savings_Annual': f"{avoided_us_co2e_kg / 1000.0:,.2f} MT",
            'Reduction_%': f"{(avoided_us_co2e_kg / baseline_us_co2e_kg) * 100:.2f}%"
        }
    ]
    
    df_ledger = pd.DataFrame(abatement_ledger)
    print("\n--- ANNUAL RESOURCE AND CARBON ABATEMENT LEDGER ---")
    print(df_ledger.to_markdown(index=False))
    
    df_ledger.to_csv(os.path.join(TABLES_DIR, "table8_carbon_resource_abatement_ledger.csv"), index=False)
    
    abatement_dict = {
        'total_annual_kwh_saved': round(total_annual_kwh_saved, 1),
        'total_annual_water_saved_liters': round(total_annual_water_saved_liters, 1),
        'avoided_india_co2e_tonnes': round(avoided_india_co2e_kg / 1000.0, 2),
        'avoided_us_co2e_tonnes': round(avoided_us_co2e_kg / 1000.0, 2),
        'energy_reduction_pct': round(energy_reduction_pct, 2),
        'water_reduction_pct': round(water_reduction_pct, 2)
    }
    with open(os.path.join(RESULTS_DIR, "carbon_abatement_metrics.json"), 'w') as f:
        json.dump(abatement_dict, f, indent=2)
        
    print(f"\n[SAVED] Table 8 -> {os.path.join(TABLES_DIR, 'table8_carbon_resource_abatement_ledger.csv')}")
    return df_ledger

if __name__ == '__main__':
    benchmark_environmental_regressors()
    calculate_carbon_and_resource_abatement()
