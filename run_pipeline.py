"""
Master Orchestration Pipeline for AI/IoT Hotel Housekeeping Management
Executes full experimental pipeline for Paper 1 (Operations/Economics) and Paper 2 (Guest/Sustainability/TAM).
"""

import os
import sys
import time

def run_step(step_name: str, module_path: str):
    print(f"\n========================================================")
    print(f"  STEP: {step_name}")
    print(f"========================================================")
    start_t = time.time()
    exit_code = os.system(f"{sys.executable} {module_path}")
    duration = time.time() - start_t
    if exit_code != 0:
        print(f"[FAILED] {step_name} failed with exit code {exit_code}")
        sys.exit(exit_code)
    print(f"[COMPLETED] {step_name} in {duration:.2f}s")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n######################################################################")
    print("  LAUNCHING FULL AI/IOT HOTEL HOUSEKEEPING RESEARCH EXPERIMENT ENGINE")
    print("######################################################################")
    
    # 1. Data Pipeline
    run_step("Data Ingestion & Cleaning", os.path.join(base_dir, "src", "data_pipeline", "fetch_data.py"))
    
    # 2. Paper 1 Modules
    run_step("Paper 1: Demand & Turnover Forecasting", os.path.join(base_dir, "src", "paper1_operations", "forecasting.py"))
    run_step("Paper 1: OR-Tools CP-SAT Scheduling Benchmark", os.path.join(base_dir, "src", "paper1_operations", "scheduler.py"))
    run_step("Paper 1: Monte Carlo Techno-Economic Simulation", os.path.join(base_dir, "src", "paper1_operations", "cost_monte_carlo.py"))
    
    # 3. Paper 2 Modules
    run_step("Paper 2: NLP Aspect Sentiment & Guest Clustering", os.path.join(base_dir, "src", "paper2_sustainability", "nlp_sentiment.py"))
    run_step("Paper 2: Calibrated Energy ML & Carbon Accounting", os.path.join(base_dir, "src", "paper2_sustainability", "environmental_models.py"))
    run_step("Paper 2: TAM Survey & Empirical Barrier Analysis", os.path.join(base_dir, "src", "paper2_sustainability", "tam_survey_analysis.py"))
    
    # 4. Publication Visualizations
    run_step("Publication 300 DPI Figure Generation", os.path.join(base_dir, "src", "visualization", "generate_figures.py"))
    
    print("\n######################################################################")
    print("  EXPERIMENT PIPELINE FULLY COMPLETED! ALL RESEARCH TABLES & FIGURES SAVED")
    print("######################################################################")
