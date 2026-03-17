"""
Baseline: Basic Bergman Minimal Model (without GI and Circadian Factors)

This is the baseline Bergman Minimal Model for comparison with the extended model.
No GI effects, no circadian rhythms, simple meal disturbance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

############################################
# 1. Data Loading and Preprocessing
############################################
def load_and_preprocess(file_path):
    """Load data, remove missing 'Libre GL' values"""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Libre GL']).fillna(0)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')
    df.sort_values('Timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Date'] = df['Timestamp'].dt.date
    return df

def preprocess_data(df):
    """Basic preprocessing"""
    df = df.copy()
    df = df.sort_values('Timestamp')
    df.reset_index(drop=True, inplace=True)
    df['Libre GL'] = df['Libre GL'].clip(40, 400)
    return df

############################################
# 2. Model Constants
############################################
Gb, Ib, n, V1, P2 = 81, 18, 5/54, 12, 0.0287

############################################
# 3. BASIC Bergman Minimal Model (NO GI, NO Circadian)
############################################
def disturb_meals_basic(t_val, meal_data, beta_meal, gamma_meal, t_lag, peak_mult, time_origin):
    """
    BASIC meal disturbance - NO GI effects, NO meal type effects
    Simple proportional to carbohydrate content only
    """
    D_total = 0
    for idx in range(len(meal_data)):
        row = meal_data.iloc[idx]
        meal_t = (row['Timestamp'] - time_origin).total_seconds() / 60.0
        
        if t_val >= meal_t + t_lag:
            dt = t_val - meal_t - t_lag
            
            # BASIC: Only carbohydrate content, NO GI multiplier, NO meal type factor
            FG = row['Carbs']  # Simple: just carbs, no GI effect
            
            # Simple dual-exponential absorption
            D_total += FG * peak_mult * (1 - np.exp(-beta_meal * dt)) * np.exp(-gamma_meal * dt)
    
    return D_total

def bergman_ode_basic(y, t_val, p1, p3, beta_meal, gamma_meal, t_lag, peak_mult, 
                      time_origin, meal_data):
    """
    BASIC Bergman Minimal Model - Original 3-equation system
    NO circadian factor, NO GI effects
    """
    G, I, X = y  # Only 3 states (no E)
    
    # BASIC meal disturbance (no GI, no meal type)
    D = disturb_meals_basic(t_val, meal_data, beta_meal, gamma_meal, t_lag, peak_mult, time_origin)
    
    # NO CIRCADIAN FACTOR - just constant dynamics
    dGdt = -p1 * G - X * (G + Gb) + D
    dIdt = -n * (I + Ib) + n * V1 * (Ib - P2 * p1 * 15 / p3 / (15 + Gb)) / V1
    dXdt = -P2 * X + p3 * I
    
    return [dGdt, dIdt, dXdt]

def solve_bergman_basic(p1, p3, beta_meal, gamma_meal, t_lag, peak_mult, 
                        time_origin, meal_data, y0, t_span):
    """Solve BASIC Bergman ODE system"""
    def ode_func(y, t_val):
        return bergman_ode_basic(y, t_val, p1, p3, beta_meal, gamma_meal, 
                                 t_lag, peak_mult, time_origin, meal_data)
    
    soln = odeint(ode_func, y0, t_span)
    G_pred = soln[:, 0] + Gb
    return G_pred

############################################
# 4. LODO Cross-Validation
############################################
def lodo_cv_split(data):
    """Leave-One-Day-Out splits"""
    sorted_dates = sorted(data['Date'].unique())
    splits = []
    for test_date in sorted_dates:
        train_dates = [d for d in sorted_dates if d != test_date]
        splits.append((train_dates, [test_date]))
    return splits

def split_train_into_5folds(train_data):
    """Split training into 5 folds"""
    sorted_dates = sorted(train_data['Date'].unique())
    n_days = len(sorted_dates)
    
    if n_days < 5:
        folds = []
        for i in range(min(n_days, 5)):
            if i < n_days:
                fold_dates = [sorted_dates[i]]
                fold_data = train_data[train_data['Date'].isin(fold_dates)].copy()
                folds.append(fold_data)
            else:
                folds.append(pd.DataFrame())
        return folds
    
    fold_size = n_days // 5
    folds = []
    for i in range(5):
        start_idx = i * fold_size
        if i == 4:
            fold_dates = sorted_dates[start_idx:]
        else:
            end_idx = start_idx + fold_size
            fold_dates = sorted_dates[start_idx:end_idx]
        
        fold_data = train_data[train_data['Date'].isin(fold_dates)].copy()
        folds.append(fold_data)
    
    return folds

############################################
# 5. Parameter Optimization
############################################
def optimize_parameters_basic(train_data, pbounds, init_points=5, n_iter=15):
    """Optimize BASIC model parameters"""
    train_data = train_data.copy()
    train_data.reset_index(drop=True, inplace=True)
    
    if len(train_data) < 10:
        return None
    
    time_origin = train_data['Timestamp'].iloc[0]
    train_meals = train_data[
        (train_data['Carbs'] > 0)  # Only need carbs, no GI
    ][['Timestamp', 'Carbs']].copy()
    
    train_data['t_min'] = (train_data['Timestamp'] - time_origin).dt.total_seconds() / 60
    t_max = train_data['t_min'].max()
    t_span = np.linspace(0, t_max, int(t_max) + 1)
    
    G0 = train_data['Libre GL'].iloc[0]
    y0 = [max(30, G0 - Gb), 0, 0]  # Only 3 states
    
    def objective(p1, p3, beta_meal, gamma_meal, t_lag, peak_mult):
        try:
            G_pred = solve_bergman_basic(
                p1, p3, beta_meal, gamma_meal, t_lag, peak_mult,
                time_origin, train_meals, y0, t_span
            )
            
            G_pred_interp = np.interp(train_data['t_min'].values, t_span, G_pred)
            rmse = np.sqrt(mean_squared_error(train_data['Libre GL'].values, G_pred_interp))
            
            return -rmse
        except:
            return -1000
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=0
    )
    
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    return optimizer.max['params']

############################################
# 6. Last-Step Prediction
############################################
def predict_last_step_basic(data, params, PH_values):
    """Last-step prediction with BASIC model"""
    data = data.copy()
    data.reset_index(drop=True, inplace=True)
    
    predictions_dict = {PH: [] for PH in PH_values}
    
    if len(data) < 2:
        return predictions_dict
    
    time_origin = data['Timestamp'].iloc[0]
    G0 = data['Libre GL'].iloc[0]
    y_current = [max(30, G0 - Gb), 0, 0]  # 3 states
    
    max_PH = max(PH_values)
    pos = 0
    max_pos = len(data) - 1
    
    while pos < max_pos:
        try:
            current_time = data.loc[pos, 'Timestamp']
            current_glucose = data.loc[pos, 'Libre GL']
            
            # Extract historical meals (only need carbs)
            historical_meals = data[
                (data['Timestamp'] <= current_time) & 
                (data['Carbs'] > 0)
            ][['Timestamp', 'Carbs']].copy()
            
            t_span = np.linspace(0, max_PH, int(max_PH) + 1)
            
            try:
                G_pred_series = solve_bergman_basic(
                    params['p1'], params['p3'], 
                    params['beta_meal'], params['gamma_meal'],
                    params['t_lag'], params['peak_mult'],
                    current_time, historical_meals, y_current, t_span
                )
            except:
                pos += 1
                continue
            
            for PH in PH_values:
                target_time = current_time + timedelta(minutes=PH)
                future_data = data[data['Timestamp'] >= target_time]
                
                if future_data.empty:
                    continue
                
                target_pos = future_data.index[0]
                if target_pos >= len(data):
                    continue
                
                actual_time = data.loc[target_pos, 'Timestamp']
                actual_glucose = data.loc[target_pos, 'Libre GL']
                
                if int(PH) < len(G_pred_series):
                    y_pred = G_pred_series[int(PH)]
                else:
                    y_pred = G_pred_series[-1]
                
                predictions_dict[PH].append({
                    'current_time': current_time,
                    'target_time': actual_time,
                    'y_true': actual_glucose,
                    'y_pred': y_pred,
                    'current_glucose': current_glucose,
                    'PH': PH
                })
            
            y_current[0] = current_glucose - Gb
            skip_interval = max(1, int(min(PH_values) / 3))
            pos += skip_interval
            
        except Exception as e:
            pos += 1
    
    return predictions_dict

############################################
# 7. Evaluation
############################################
def evaluate_predictions(predictions_list):
    """Calculate metrics"""
    if not predictions_list:
        return None
    
    y_true = np.array([p['y_true'] for p in predictions_list])
    y_pred = np.array([p['y_pred'] for p in predictions_list])
    y_pred = np.clip(y_pred, 40, 400)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    time_in_range = np.mean((y_pred >= 70) & (y_pred <= 180)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Time_in_Range': time_in_range,
        'N_Predictions': len(predictions_list)
    }

############################################
# 8. Main Function
############################################
def main(file_path, prediction_horizons=[15, 30, 45, 60], 
         output_dir='results_baseline_bergman'):
    """
    BASELINE: Basic Bergman Minimal Model
    - No GI effects
    - No circadian rhythms
    - No meal type differentiation
    - Simple carbohydrate-based meal disturbance
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    print("=" * 80)
    print("BASELINE: BASIC BERGMAN MINIMAL MODEL (No GI, No Circadian)")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    try:
        data = load_and_preprocess(file_path)
        data = preprocess_data(data)
    except Exception as e:
        print(f"Error: {e}")
        return None, None
    
    unique_dates = sorted(data['Date'].unique())
    print(f"Total measurements: {len(data)}")
    print(f"Total days: {len(unique_dates)}")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    
    if len(unique_dates) < 3:
        print("Error: Need at least 3 days")
        return None, None
    
    # LODO splits
    print("\nCreating LODO splits...")
    lodo_splits = lodo_cv_split(data)
    print(f"Number of LODO folds: {len(lodo_splits)}")
    
    # Parameter bounds (same 6 parameters)
    pbounds = {
        'p1': (0.005, 0.05),
        'p3': (1e-6, 1e-4),
        'beta_meal': (0.01, 0.5),
        'gamma_meal': (0.01, 0.5),
        't_lag': (10, 60),
        'peak_mult': (2.0, 6.0)
    }
    
    all_results = {PH: [] for PH in prediction_horizons}
    
    # LODO Loop
    print("\n" + "=" * 80)
    print("LODO Cross-Validation with Internal 5-Fold")
    print("=" * 80)
    
    for lodo_idx, (train_dates, test_dates) in enumerate(lodo_splits):
        test_date = test_dates[0]
        
        print(f"\n{'='*60}")
        print(f"LODO Fold {lodo_idx + 1}/{len(lodo_splits)}")
        print(f"Test date: {test_date}")
        print(f"{'='*60}")
        
        train_data = data[data['Date'].isin(train_dates)].copy()
        test_data = data[data['Date'].isin(test_dates)].copy()
        
        if len(train_data) < 10 or len(test_data) < 5:
            print("Insufficient data, skipping")
            continue
        
        # Internal 5-fold split
        internal_folds = split_train_into_5folds(train_data)
        
        # Optimize on Internal Fold 1
        print("  Optimizing parameters on Fold 1...")
        if len(internal_folds[0]) < 10:
            print("  Insufficient data, skipping")
            continue
        
        best_params = optimize_parameters_basic(internal_folds[0], pbounds, 
                                               init_points=3, n_iter=10)
        
        if best_params is None:
            print("  Optimization failed, skipping")
            continue
        
        print("  Optimized parameters:")
        for key, val in best_params.items():
            print(f"    {key:15s}: {val:.6f}")
        
        # Test on held-out day
        print(f"  Testing on {test_date}...")
        test_predictions = predict_last_step_basic(test_data, best_params, 
                                                   prediction_horizons)
        
        for PH in prediction_horizons:
            if test_predictions[PH]:
                metrics = evaluate_predictions(test_predictions[PH])
                
                if metrics:
                    print(f"    PH={PH}min: RMSE={metrics['RMSE']:.2f} mg/dL, "
                          f"N={metrics['N_Predictions']}")
                    
                    all_results[PH].append({
                        'lodo_fold': lodo_idx + 1,
                        'test_date': test_date,
                        **metrics,
                        **best_params
                    })
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("FINAL RESULTS - BASELINE MODEL")
    print("=" * 80)
    
    summary_results = []
    
    for PH in prediction_horizons:
        if not all_results[PH]:
            continue
        
        results_df = pd.DataFrame(all_results[PH])
        
        rmse_mean = results_df['RMSE'].mean()
        rmse_std = results_df['RMSE'].std()
        mae_mean = results_df['MAE'].mean()
        mae_std = results_df['MAE'].std()
        
        print(f"\n{'='*60}")
        print(f"PH: {PH} minutes")
        print(f"{'='*60}")
        print(f"RMSE: {rmse_mean:7.2f} ± {rmse_std:6.2f} mg/dL")
        print(f"MAE:  {mae_mean:7.2f} ± {mae_std:6.2f} mg/dL")
        print(f"N folds: {len(results_df)}")
        
        summary_results.append({
            'PH': PH,
            'RMSE_mean': rmse_mean,
            'RMSE_std': rmse_std,
            'MAE_mean': mae_mean,
            'MAE_std': mae_std,
            'MAPE_mean': results_df['MAPE'].mean(),
            'MAPE_std': results_df['MAPE'].std(),
            'Time_in_Range_mean': results_df['Time_in_Range'].mean(),
            'Time_in_Range_std': results_df['Time_in_Range'].std(),
            'N_folds': len(results_df)
        })
        
        results_df.to_csv(
            os.path.join(output_dir, f'baseline_detailed_PH{PH}.csv'),
            index=False
        )
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(os.path.join(output_dir, 'baseline_summary.csv'), 
                         index=False)
        
        print("\n" + "=" * 80)
        print("SUMMARY TABLE - BASELINE BERGMAN MODEL")
        print("=" * 80)
        print("\nPH (min) | RMSE (mg/dL)    | MAE (mg/dL)")
        print("-" * 50)
        for _, row in summary_df.iterrows():
            print(f"{int(row['PH']):8d} | {row['RMSE_mean']:6.2f} ± {row['RMSE_std']:5.2f} | "
                  f"{row['MAE_mean']:6.2f} ± {row['MAE_std']:5.2f}")
        
        print("\n" + "=" * 80)
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
        print("\nModel Features:")
        print("  ✓ Basic Bergman Minimal Model (3 ODE equations)")
        print("  ✗ No GI effects")
        print("  ✗ No circadian rhythms")
        print("  ✗ No meal type differentiation")
        print("  ✓ Simple carbohydrate-based meal disturbance")
        print("=" * 80)
        
        return summary_df, all_results
    else:
        print("\nNo results generated!")
        return None, None

############################################
# CLI
############################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline Bergman Minimal Model runner")
    parser.add_argument("--input_csv", required=True, help="Path to one subject CSV file")
    parser.add_argument("--output_dir", default="results_baseline", help="Directory to save outputs")
    parser.add_argument("--prediction_horizons", nargs="+", type=int, default=[15, 30, 45, 60], help="Prediction horizons in minutes")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_df, all_results = main(
        file_path=args.input_csv,
        prediction_horizons=args.prediction_horizons,
        output_dir=args.output_dir,
    )
