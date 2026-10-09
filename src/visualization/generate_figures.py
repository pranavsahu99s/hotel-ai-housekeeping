"""
Visualization Module for Journal Publication
Generates 300 DPI publication-grade figures (PNG and PDF) for Paper 1 and Paper 2.
Adheres to IEEE/Elsevier visual standards: clear typography, colorblind-friendly palettes, and multi-panel layouts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set publication plotting aesthetic
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.35
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_P1 = os.path.join(BASE_DIR, "results", "paper1")
RESULTS_P2 = os.path.join(BASE_DIR, "results", "paper2")
FIG_P1_DIR = os.path.join(RESULTS_P1, "figures")
FIG_P2_DIR = os.path.join(RESULTS_P2, "figures")
os.makedirs(FIG_P1_DIR, exist_ok=True)
os.makedirs(FIG_P2_DIR, exist_ok=True)

def generate_paper1_figures():
    print("=== [VISUALIZATION] GENERATING 300 DPI FIGURES FOR PAPER 1 ===")
    
    # FIGURE 1: Forecasting Benchmark Time-Series & Parity
    preds_file = os.path.join(RESULTS_P1, "test_predictions_time_series.csv")
    if os.path.exists(preds_file):
        df_pred = pd.read_csv(preds_file)
        df_pred['date'] = pd.to_datetime(df_pred['date'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, gridspec_kw={'width_ratios': [1.8, 1.2]})
        
        # Panel A: Time Series (Subset 90 days for clarity)
        sub = df_pred.iloc[:90]
        axes[0].plot(sub['date'], sub['actual_workload'], label='Ground Truth Workload', color='black', linewidth=1.8)
        axes[0].plot(sub['date'], sub['pred_ma7'], label='7D Moving Average', color='#e74c3c', linestyle='--', alpha=0.7)
        axes[0].plot(sub['date'], sub['pred_lightgbm'], label='LightGBM Forecaster', color='#2980b9', linewidth=1.5)
        axes[0].set_title('(a) Out-of-Time Workload Forecast Trajectory (Test Period: 2017)', fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Daily Turnover Tasks Required')
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        axes[0].legend(frameon=True, facecolor='white', loc='upper right')
        
        # Panel B: Parity Scatter Plot (Actual vs LightGBM)
        axes[1].scatter(df_pred['actual_workload'], df_pred['pred_lightgbm'], alpha=0.45, color='#2980b9', edgecolors='none', s=25)
        min_val = min(df_pred['actual_workload'].min(), df_pred['pred_lightgbm'].min())
        max_val = max(df_pred['actual_workload'].max(), df_pred['pred_lightgbm'].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Parity (1:1)')
        axes[1].set_title('(b) Model Parity: Actual vs. LightGBM', fontweight='bold')
        axes[1].set_xlabel('Actual Cleaning Tasks')
        axes[1].set_ylabel('Predicted Cleaning Tasks')
        axes[1].legend(frameon=True, facecolor='white')
        
        plt.tight_layout()
        fig1_path = os.path.join(FIG_P1_DIR, "figure1_forecasting_benchmark.png")
        plt.savefig(fig1_path, dpi=300)
        plt.savefig(fig1_path.replace('.png', '.pdf'))
        plt.close()
        print(f"  [SAVED] Figure 1 -> {fig1_path}")
        
    # FIGURE 2: Operational Simulation & Turnaround Times
    trace_base_file = os.path.join(RESULTS_P1, "simulation_traces_baseline.csv")
    trace_cp_file = os.path.join(RESULTS_P1, "simulation_traces_cpsat.csv")
    if os.path.exists(trace_base_file) and os.path.exists(trace_cp_file):
        df_b = pd.read_csv(trace_base_file)
        df_c = pd.read_csv(trace_cp_file)
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # Panel A: Boxplot comparison of Makespan & Turnaround
        comp_data = pd.DataFrame({
            'Makespan (Greedy Baseline)': df_b['makespan_min'],
            'Makespan (OR-Tools CP-SAT)': df_c['makespan_min'],
            'Turnaround (Baseline)': df_b['avg_turnaround_time_min'],
            'Turnaround (CP-SAT)': df_c['avg_turnaround_time_min']
        })
        sns.boxplot(data=comp_data, ax=axes[0], palette=['#e74c3c', '#27ae60', '#f39c12', '#2980b9'])
        axes[0].set_title('(a) Floor Makespan and Turnaround Time Distributions', fontweight='bold')
        axes[0].set_ylabel('Minutes')
        axes[0].tick_params(axis='x', rotation=20)
        
        # Panel B: Overtime Hours Saved Distribution
        ot_data = pd.DataFrame({
            'Greedy Baseline': df_b['total_overtime_min'] / 60.0,
            'OR-Tools CP-SAT': df_c['total_overtime_min'] / 60.0
        })
        sns.barplot(data=ot_data, ax=axes[1], errorbar='sd', palette=['#e74c3c', '#27ae60'], capsize=0.1)
        axes[1].set_title('(b) Total Shift Overtime Incurred (Hours/Day ± Std)', fontweight='bold')
        axes[1].set_ylabel('Overtime Hours per Shift')
        
        plt.tight_layout()
        fig2_path = os.path.join(FIG_P1_DIR, "figure2_scheduling_operational_comparison.png")
        plt.savefig(fig2_path, dpi=300)
        plt.savefig(fig2_path.replace('.png', '.pdf'))
        plt.close()
        print(f"  [SAVED] Figure 2 -> {fig2_path}")
        
    # FIGURE 3: Monte Carlo CPOR Distributions & Payback Frontier
    mc_sample_file = os.path.join(RESULTS_P1, "monte_carlo_distribution_sample_india.csv")
    if os.path.exists(mc_sample_file):
        df_mc = pd.read_csv(mc_sample_file)
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # Panel A: CPOR Distribution Density
        sns.kdeplot(df_mc['baseline_cpor'], ax=axes[0], label='Baseline Operations', color='#e74c3c', fill=True, alpha=0.3)
        sns.kdeplot(df_mc['ai_cpor'], ax=axes[0], label='AI/IoT Optimized', color='#27ae60', fill=True, alpha=0.3)
        axes[0].axvline(df_mc['baseline_cpor'].mean(), color='#e74c3c', linestyle='--')
        axes[0].axvline(df_mc['ai_cpor'].mean(), color='#27ae60', linestyle='--')
        axes[0].set_title('(a) Stochastic Cost Per Occupied Room (CPOR) Density', fontweight='bold')
        axes[0].set_xlabel('Cost Per Occupied Room (₹ INR / room-night)')
        axes[0].set_ylabel('Probability Density')
        axes[0].legend(frameon=True, facecolor='white')
        
        # Panel B: Payback Period Distribution (Months)
        sns.histplot(df_mc['payback_months'], ax=axes[1], bins=35, kde=True, color='#2980b9', edgecolor='black')
        axes[1].axvline(df_mc['payback_months'].median(), color='red', linestyle='--', label=f'Median Breakeven: {df_mc["payback_months"].median():.1f} mo')
        axes[1].set_title('(b) Capital Investment Payback Horizon (120 Rooms)', fontweight='bold')
        axes[1].set_xlabel('Months to Capital Amortization')
        axes[1].set_ylabel('Simulation Iteration Count')
        axes[1].legend(frameon=True, facecolor='white')
        
        plt.tight_layout()
        fig3_path = os.path.join(FIG_P1_DIR, "figure3_monte_carlo_cpor_payback.png")
        plt.savefig(fig3_path, dpi=300)
        plt.savefig(fig3_path.replace('.png', '.pdf'))
        plt.close()
        print(f"  [SAVED] Figure 3 -> {fig3_path}")

def generate_paper2_figures():
    print("=== [VISUALIZATION] GENERATING 300 DPI FIGURES FOR PAPER 2 ===")
    
    # FIGURE 4: Aspect-Based Sentiment Radar Plot
    aspect_file = os.path.join(RESULTS_P2, "aspect_polarity_distributions.json")
    if os.path.exists(aspect_file):
        with open(aspect_file, 'r') as f:
            aspect_data = json.load(f)
            
        categories = ['Cleanliness\n& Dust', 'Bathroom\nPlumbing', 'Linen\n& Bedding', 'Odor &\nFreshness', 'Promptness\n& Service']
        # Convert means to normalized radar scores (0 to 1 scale)
        values = [max(0.2, (v + 1.0) / 2.0) for v in aspect_data.values()]
        values += values[:1] # Close polygon
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)
        ax.plot(angles, values, color='#2980b9', linewidth=2, linestyle='solid')
        ax.fill(angles, values, color='#3498db', alpha=0.35)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title('Housekeeping Aspect-Based Sentiment Profile\n(TripAdvisor Corpus)', y=1.08, fontweight='bold')
        
        fig4_path = os.path.join(FIG_P2_DIR, "figure4_aspect_sentiment_radar.png")
        plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig4_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  [SAVED] Figure 4 -> {fig4_path}")
        
    # FIGURE 5: Carbon & Resource Abatement Ledger
    abatement_file = os.path.join(RESULTS_P2, "carbon_abatement_metrics.json")
    if os.path.exists(abatement_file):
        with open(abatement_file, 'r') as f:
            abatement = json.load(f)
            
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # Panel A: Electricity & Water Savings %
        metrics = ['Electricity Savings', 'Water Savings']
        pcts = [abatement['energy_reduction_pct'], abatement['water_reduction_pct']]
        bars = axes[0].bar(metrics, pcts, color=['#f39c12', '#2980b9'], width=0.45, edgecolor='black')
        axes[0].set_ylabel('Percentage Reduction vs. Traditional Baseline (%)')
        axes[0].set_title('(a) Resource Conservation Efficiency', fontweight='bold')
        axes[0].set_ylim(0, max(pcts) * 1.3)
        for bar in bars:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
            
        # Panel B: Avoided Carbon Emissions (India vs US Grids)
        grids = ['India Grid (CEA)', 'US Grid (EPA eGRID)']
        co2_vals = [abatement['avoided_india_co2e_tonnes'], abatement['avoided_us_co2e_tonnes']]
        bars2 = axes[1].bar(grids, co2_vals, color=['#27ae60', '#16a085'], width=0.45, edgecolor='black')
        axes[1].set_ylabel('Avoided Carbon (Metric Tonnes CO2e / Year)')
        axes[1].set_title('(b) Annual Carbon Abatement (120 Rooms)', fontweight='bold')
        axes[1].set_ylim(0, max(co2_vals) * 1.25)
        for bar in bars2:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f} MT", ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        fig5_path = os.path.join(FIG_P2_DIR, "figure5_carbon_resource_abatement.png")
        plt.savefig(fig5_path, dpi=300)
        plt.savefig(fig5_path.replace('.png', '.pdf'))
        plt.close()
        print(f"  [SAVED] Figure 5 -> {fig5_path}")
        
    # FIGURE 6: TAM Adoption Barrier Hierarchy
    barrier_file = os.path.join(RESULTS_P2, "tables", "table10_adoption_barrier_hierarchy.csv")
    if os.path.exists(barrier_file):
        df_bar = pd.read_csv(barrier_file)
        
        plt.figure(figsize=(9, 5), dpi=300)
        y_pos = np.arange(len(df_bar))
        colors = ['#c0392b' if tier == 'Tier 1 (Severe Roadblock)' else '#e67e22' for tier in df_bar['Criticality_Tier']]
        
        plt.barh(y_pos, df_bar['RII'], color=colors, edgecolor='black', height=0.6)
        plt.yticks(y_pos, df_bar['Barrier_Description'])
        plt.gca().invert_yaxis()
        plt.xlabel('Relative Importance Index (RII)')
        plt.title('Empirical Hospitality Technology Adoption Barriers (TAM/UTAUT)', fontweight='bold')
        plt.xlim(0.4, 1.0)
        
        for i, v in enumerate(df_bar['RII']):
            plt.text(v + 0.01, i, f"RII={v:.3f}", va='center', fontsize=9, fontweight='bold')
            
        plt.tight_layout()
        fig6_path = os.path.join(FIG_P2_DIR, "figure6_tam_barrier_hierarchy.png")
        plt.savefig(fig6_path, dpi=300)
        plt.savefig(fig6_path.replace('.png', '.pdf'))
        plt.close()
        print(f"  [SAVED] Figure 6 -> {fig6_path}")

if __name__ == '__main__':
    generate_paper1_figures()
    generate_paper2_figures()
