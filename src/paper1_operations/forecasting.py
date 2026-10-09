"""
Paper 1 Module: Predictive Demand & Workload Forecasting
Compares Classical Baselines (Moving Average, Ridge Lag) vs. Gradient-Boosted Trees (LightGBM, XGBoost)
Uses strict chronological train/test split to prevent temporal data leakage.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
try:
    import xgboost as xgb
except ImportError:
    xgb = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "paper1")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def load_and_engineer_features():
    workload_path = os.path.join(PROCESSED_DIR, "daily_turnover_workload.csv")
    if not os.path.exists(workload_path):
        raise FileNotFoundError(f"Workload dataset not found at {workload_path}. Run fetch_data.py first.")
    
    df = pd.read_csv(workload_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Target variable: total cleaning tasks (checkout turnovers + stayovers)
    y_col = 'total_cleaning_tasks'
    
    # Feature Engineering: Lagged operational workload
    for lag in [1, 2, 7, 14, 21, 28]:
        df[f'lag_{lag}'] = df[y_col].shift(lag)
        df[f'checkout_lag_{lag}'] = df['checkout_count'].shift(lag)
        df[f'stayover_lag_{lag}'] = df['stayover_count'].shift(lag)
        
    # Rolling window statistics
    df['rolling_mean_7'] = df[y_col].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df[y_col].shift(1).rolling(window=7).std()
    df['rolling_mean_14'] = df[y_col].shift(1).rolling(window=14).mean()
    df['rolling_mean_28'] = df[y_col].shift(1).rolling(window=28).mean()
    
    # Calendar features
    df['day_of_week_num'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month_num'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    
    # Drop rows with NaN from lagging (first 28 days)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean, y_col

def train_and_benchmark_forecasting():
    print("=== [PAPER 1] WORKLOAD DEMAND FORECASTING BENCHMARK ===")
    df, y_col = load_and_engineer_features()
    
    # Chronological Split: Train on 2015-2016, Test on 2017
    train_mask = df['date'] < '2017-01-01'
    test_mask = df['date'] >= '2017-01-01'
    
    feature_cols = [
        col for col in df.columns if col not in [
            'date', 'day_of_week', 'total_cleaning_tasks', 
            'checkout_count', 'stayover_count', 'checkout_suites', 
            'checkout_standard', 'stayover_suites', 'stayover_standard'
        ]
    ]
    
    X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, y_col]
    X_test, y_test = df.loc[test_mask, feature_cols], df.loc[test_mask, y_col]
    test_dates = df.loc[test_mask, 'date']
    
    print(f"Dataset split: Train = {len(X_train)} days, Test = {len(X_test)} days across {len(feature_cols)} features.")
    
    results = {}
    predictions = {'date': test_dates.dt.strftime('%Y-%m-%d').values, 'actual_workload': y_test.values}
    
    # 1. Baseline: 7-day Moving Average
    pred_ma7 = df.loc[test_mask, 'rolling_mean_7'].values
    mae_ma = mean_absolute_error(y_test, pred_ma7)
    rmse_ma = np.sqrt(mean_squared_error(y_test, pred_ma7))
    r2_ma = r2_score(y_test, pred_ma7)
    mape_ma = np.mean(np.abs((y_test - pred_ma7) / np.maximum(y_test, 1))) * 100
    results['Moving_Average_7D'] = {
        'MAE': round(mae_ma, 2), 'RMSE': round(rmse_ma, 2), 
        'MAPE_%': round(mape_ma, 2), 'R2': round(r2_ma, 4)
    }
    predictions['pred_ma7'] = pred_ma7
    
    # 2. Baseline: Ridge Regression (Classical Linear Benchmark)
    ridge = Ridge(alpha=10.0, random_state=42)
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)
    results['Ridge_Regression'] = {
        'MAE': round(mean_absolute_error(y_test, pred_ridge), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, pred_ridge)), 2),
        'MAPE_%': round(np.mean(np.abs((y_test - pred_ridge) / np.maximum(y_test, 1))) * 100, 2),
        'R2': round(r2_score(y_test, pred_ridge), 4)
    }
    predictions['pred_ridge'] = pred_ridge
    
    # 3. Main Model: LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150, learning_rate=0.04, num_leaves=31,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    pred_lgb = lgb_model.predict(X_test)
    results['LightGBM'] = {
        'MAE': round(mean_absolute_error(y_test, pred_lgb), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, pred_lgb)), 2),
        'MAPE_%': round(np.mean(np.abs((y_test - pred_lgb) / np.maximum(y_test, 1))) * 100, 2),
        'R2': round(r2_score(y_test, pred_lgb), 4)
    }
    predictions['pred_lightgbm'] = pred_lgb
    
    # 4. Main Model: XGBoost Regressor (or GradientBoostingRegressor fallback)
    if xgb is not None:
        xgb_model = xgb.XGBRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=5,
            random_state=42, objective='reg:squarederror'
        )
        xgb_model.fit(X_train, y_train)
        pred_xgb = xgb_model.predict(X_test)
        model_name = 'XGBoost'
    else:
        gb_model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=5,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        pred_xgb = gb_model.predict(X_test)
        model_name = 'GradientBoosting'
        
    results[model_name] = {
        'MAE': round(mean_absolute_error(y_test, pred_xgb), 2),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, pred_xgb)), 2),
        'MAPE_%': round(np.mean(np.abs((y_test - pred_xgb) / np.maximum(y_test, 1))) * 100, 2),
        'R2': round(r2_score(y_test, pred_xgb), 4)
    }
    predictions['pred_gradient_boosting'] = pred_xgb
    
    # Print comparison table
    df_results = pd.DataFrame(results).T
    print("\n--- FORECASTING MODEL BENCHMARK RESULTS ---")
    print(df_results.to_markdown())
    
    # Save outputs
    metrics_json_path = os.path.join(RESULTS_DIR, "forecasting_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    df_results.to_csv(os.path.join(TABLES_DIR, "table1_forecasting_benchmark.csv"))
    pd.DataFrame(predictions).to_csv(os.path.join(RESULTS_DIR, "test_predictions_time_series.csv"), index=False)
    
    # Feature Importance for Explainability (XAI)
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_imp.to_csv(os.path.join(TABLES_DIR, "forecasting_feature_importance.csv"), index=False)
    
    print(f"\n[SAVED] Benchmark metrics -> {metrics_json_path}")
    print(f"[SAVED] Publication table -> {os.path.join(TABLES_DIR, 'table1_forecasting_benchmark.csv')}")
    return results, df_results

if __name__ == '__main__':
    train_and_benchmark_forecasting()
