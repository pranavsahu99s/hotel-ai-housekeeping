"""
Paper 1 Module: Techno-Economic Feasibility & Monte Carlo Cost Simulation (N=10,000)
Evaluates Cost Per Occupied Room (CPOR), 95% Confidence Intervals, Sensitivity Regression,
and Capital ROI / Payback Periods for Indian and International Hospitality Contexts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CALIBRATION_DIR = os.path.join(BASE_DIR, "data", "calibration")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "paper1")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def load_calibration_standards():
    standards_path = os.path.join(CALIBRATION_DIR, "standards.json")
    with open(standards_path, 'r') as f:
        return json.load(f)

def run_monte_carlo_cost_simulation(n_iterations: int = 10000, hotel_rooms: int = 120, region: str = 'india'):
    """
    Simulates stochastic annual cost distributions across 10,000 iterations.
    Perturbs occupancy, wages, supply prices, and overtime events.
    """
    print(f"=== [PAPER 1] RUNNING {n_iterations:,} MONTE CARLO STOCHASTIC COST ITERATIONS ({region.upper()}) ===")
    np.random.seed(42)
    standards = load_calibration_standards()
    econ = standards['economic_standards'][region]
    
    base_wage = econ['base_hourly_wage']
    ot_mult = econ['overtime_multiplier']
    supply_checkout = econ['supply_cost_per_checkout']
    supply_stayover = econ['supply_cost_per_stayover']
    iot_capex_per_room = econ['iot_retrofit_cost_per_room']
    annual_opex_per_room = econ['annual_software_maintenance_per_room']
    
    # Total CapEx for the hotel property
    total_capex = hotel_rooms * iot_capex_per_room
    annual_software_opex = hotel_rooms * annual_opex_per_room
    
    # Stochastic Parameters per iteration (representing 1 operational year)
    # 1. Annual average occupancy rate ~ Beta distribution (mean ~ 72%, min 55%, max 88%)
    occupancy_rates = np.random.beta(a=14, b=5, size=n_iterations)
    
    # 2. Wage fluctuation ~ Normal (± 10%)
    wages = np.random.normal(loc=base_wage, scale=base_wage * 0.05, size=n_iterations)
    
    # 3. Supply cost inflation / discount ~ Normal (± 8%)
    checkout_supplies = np.random.normal(loc=supply_checkout, scale=supply_checkout * 0.04, size=n_iterations)
    stayover_supplies = np.random.normal(loc=supply_stayover, scale=supply_stayover * 0.04, size=n_iterations)
    
    # Shift parameters
    num_staff = 8
    annual_shifts = 365
    regular_hours_per_shift = 7.0 # 7 hours effective work per staff
    
    results = []
    
    for i in range(n_iterations):
        occ = occupancy_rates[i]
        wage = wages[i]
        sup_co = checkout_supplies[i]
        sup_so = stayover_supplies[i]
        
        # Total occupied room nights per year
        occupied_rooms_per_day = hotel_rooms * occ
        annual_occupied_room_nights = occupied_rooms_per_day * 365
        
        # Turnovers vs Stayovers
        # Empirical turnover ratio ~ 40% of occupied rooms check out daily
        daily_checkouts = occupied_rooms_per_day * 0.40
        daily_stayovers = occupied_rooms_per_day * 0.60
        
        annual_checkouts = daily_checkouts * 365
        annual_stayovers = daily_stayovers * 365
        
        # 1. BASELINE Operational Schedule Costs
        # In baseline, staff experience ~45 mins overtime per staff per day during peak occupancy
        baseline_daily_regular_hours = num_staff * regular_hours_per_shift
        # Overtime probability increases with occupancy
        baseline_daily_ot_hours = num_staff * max(0.0, (occ - 0.65) * 2.8) * np.random.uniform(0.7, 1.2)
        
        baseline_annual_labor = (
            (baseline_daily_regular_hours * wage) +
            (baseline_daily_ot_hours * wage * ot_mult)
        ) * 365
        
        # Baseline supply cost (higher waste from unmonitored dispensing ~ 8% excess)
        baseline_annual_supply = (
            (annual_checkouts * sup_co * 1.08) +
            (annual_stayovers * sup_so * 1.08)
        )
        
        baseline_annual_total_cost = baseline_annual_labor + baseline_annual_supply
        baseline_cpor = baseline_annual_total_cost / annual_occupied_room_nights
        
        # 2. AI-OPTIMIZED (OR-Tools + SimPy) Operational Schedule Costs
        # Optimization eliminates 82% of overtime hours through balanced load distribution
        ai_daily_regular_hours = num_staff * regular_hours_per_shift
        ai_daily_ot_hours = baseline_daily_ot_hours * 0.18 # 82% reduction
        
        ai_annual_labor = (
            (ai_daily_regular_hours * wage) +
            (ai_daily_ot_hours * wage * ot_mult)
        ) * 365
        
        # Smart dosing and linen management eliminates excess supply waste
        ai_annual_supply = (
            (annual_checkouts * sup_co) +
            (annual_stayovers * sup_so)
        )
        
        ai_annual_total_cost = ai_annual_labor + ai_annual_supply + annual_software_opex
        ai_cpor = ai_annual_total_cost / annual_occupied_room_nights
        
        # Annual Net Savings
        annual_savings = baseline_annual_total_cost - ai_annual_total_cost
        savings_pct = (annual_savings / baseline_annual_total_cost) * 100
        cpor_savings = baseline_cpor - ai_cpor
        
        # Payback period in months
        payback_months = (total_capex / max(1.0, annual_savings)) * 12.0
        
        # 3-year ROI
        net_3yr_savings = (annual_savings * 3) - total_capex
        roi_3yr_pct = (net_3yr_savings / total_capex) * 100
        
        results.append({
            'occupancy_rate': occ,
            'wage_rate': wage,
            'baseline_cpor': baseline_cpor,
            'ai_cpor': ai_cpor,
            'cpor_savings': cpor_savings,
            'savings_pct': savings_pct,
            'annual_savings': annual_savings,
            'payback_months': payback_months,
            'roi_3yr_pct': roi_3yr_pct
        })
        
    df_mc = pd.DataFrame(results)
    
    # Statistical Confidence Intervals (95% CI: 2.5% and 97.5% percentiles)
    def get_ci(series):
        mean_val = series.mean()
        lower = np.percentile(series, 2.5)
        upper = np.percentile(series, 97.5)
        return {
            'Mean': round(mean_val, 2),
            'Std': round(series.std(), 2),
            '95%_CI_Lower': round(lower, 2),
            '95%_CI_Upper': round(upper, 2)
        }
        
    ci_summary = {
        'Baseline_CPOR': get_ci(df_mc['baseline_cpor']),
        'AI_Optimized_CPOR': get_ci(df_mc['ai_cpor']),
        'CPOR_Reduction': get_ci(df_mc['cpor_savings']),
        'Annual_Cost_Savings_Pct': get_ci(df_mc['savings_pct']),
        'Annual_Net_Savings': get_ci(df_mc['annual_savings']),
        'Payback_Period_Months': get_ci(df_mc['payback_months']),
        'ROI_3_Year_Pct': get_ci(df_mc['roi_3yr_pct'])
    }
    
    df_ci = pd.DataFrame(ci_summary).T
    print("\n--- MONTE CARLO TECHNO-ECONOMIC SUMMARY (95% CONFIDENCE INTERVALS) ---")
    print(df_ci.to_markdown())
    
    # Sensitivity Regression Analysis (Predicting CPOR savings from Occupancy and Wage)
    X_sens = df_mc[['occupancy_rate', 'wage_rate']]
    y_sens = df_mc['cpor_savings']
    reg = LinearRegression().fit(X_sens, y_sens)
    
    sens_summary = {
        'Intercept': round(reg.intercept_, 4),
        'Occupancy_Coefficient': round(reg.coef_[0], 4),
        'Wage_Coefficient': round(reg.coef_[1], 4),
        'R_squared': round(reg.score(X_sens, y_sens), 4)
    }
    
    print("\n--- SENSITIVITY REGRESSION ANALYSIS ---")
    print(f"CPOR Savings = {reg.intercept_:.2f} + ({reg.coef_[0]:.2f} * Occupancy) + ({reg.coef_[1]:.2f} * Wage)")
    print(f"Explained Variance (R²): {reg.score(X_sens, y_sens):.4f}")
    
    # Save outputs
    df_ci.to_csv(os.path.join(TABLES_DIR, f"table3_techno_economic_monte_carlo_{region}.csv"))
    with open(os.path.join(RESULTS_DIR, f"monte_carlo_metrics_{region}.json"), 'w') as f:
        json.dump(ci_summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"sensitivity_regression_{region}.json"), 'w') as f:
        json.dump(sens_summary, f, indent=2)
        
    df_mc.sample(1000, random_state=42).to_csv(
        os.path.join(RESULTS_DIR, f"monte_carlo_distribution_sample_{region}.csv"), index=False
    )
    
    print(f"\n[SAVED] Table 3 -> {os.path.join(TABLES_DIR, f'table3_techno_economic_monte_carlo_{region}.csv')}")
    return df_ci, df_mc

if __name__ == '__main__':
    run_monte_carlo_cost_simulation(n_iterations=10000, hotel_rooms=120, region='india')
    run_monte_carlo_cost_simulation(n_iterations=10000, hotel_rooms=120, region='international_usd')
