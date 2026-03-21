"""
Blood Glucose Prediction using Extended Bergman Model
LODO Cross-Validation with Internal 5-Fold Processing (No Data Leakage)

改进版LODO：
- 每个LODO fold的训练集内部再分5折
- Fold 1: 参数优化
- Fold 2-4: 异常检测
- Fold 5: 内部验证
- 完全避免数据泄露
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
    """Enhanced data preprocessing"""
    df = df.copy()
    df = df.sort_values('Timestamp')
    df.reset_index(drop=True, inplace=True)

    windows = [3, 6, 12]
    for window in windows:
        df[f'GL_rolling_mean_{window}'] = df['Libre GL'].rolling(window=window, min_periods=1).mean()
        df[f'GL_rolling_std_{window}'] = df['Libre GL'].rolling(window=window, min_periods=1).std()

    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['Libre GL'] = df['Libre GL'].clip(40, 400)
    df['GL_diff'] = df['Libre GL'].diff()
    df['GL_diff_rate'] = df['GL_diff'] / df['Libre GL'].shift(1)
    df.fillna(0, inplace=True)

    return df


############################################
# 2. Model Constants
############################################
Gb, Ib, n, V1, P2 = 81, 18, 5 / 54, 12, 0.0287


############################################
# 3. Extended Bergman Model
############################################
def disturb_meals(t_val, meal_data, beta_meal, gamma_meal, t_lag, peak_mult, time_origin):
    """Calculate meal disturbance"""
    D_total = 0
    for idx in range(len(meal_data)):
        row = meal_data.iloc[idx]
        meal_t = (row['Timestamp'] - time_origin).total_seconds() / 60.0

        if t_val >= meal_t + t_lag:
            dt = t_val - meal_t - t_lag

            if row['GI'] > 70:
                gi_multiplier = 1.2 + (row['GI'] - 70) * 0.005
            elif row['GI'] < 30:
                gi_multiplier = 0.8 - (30 - row['GI']) * 0.005
            else:
                gi_multiplier = 1.0 + (row['GI'] - 50) * 0.004

            meal_type_factor = 1.0
            meal_type_lower = str(row['Meal Type']).lower()
            if 'breakfast' in meal_type_lower:
                meal_type_factor = 1.2
            elif 'dinner' in meal_type_lower:
                meal_type_factor = 0.9

            FG = row['Carbs'] * (row['GI'] / 100.0) * gi_multiplier * meal_type_factor
            D_total += FG * peak_mult * (1 - np.exp(-beta_meal * dt)) * np.exp(-gamma_meal * dt)

    return D_total


def bergman_ode(y, t_val, p1, p3, beta_meal, gamma_meal, t_lag, peak_mult,
                time_origin, meal_data):
    """Extended Bergman minimal model"""
    G, I, X, E = y

    D = disturb_meals(t_val, meal_data, beta_meal, gamma_meal, t_lag, peak_mult, time_origin)

    hour_of_day = (t_val / 60) % 24
    morning_effect = np.exp(-((hour_of_day - 6) ** 2) / 8)
    evening_effect = np.exp(-((hour_of_day - 18) ** 2) / 16)
    circadian_factor = 1 + 0.15 * morning_effect - 0.1 * evening_effect

    dEdt = np.clip((G - 15), -15, 15)
    dGdt = (-p1 * G - X * (G + Gb) + D) * circadian_factor
    dIdt = -n * (I + Ib) + n * V1 * (Ib - P2 * p1 * 15 / p3 / (15 + Gb)) / V1
    dXdt = -P2 * X + p3 * I

    return [dGdt, dIdt, dXdt, dEdt]


def solve_bergman(p1, p3, beta_meal, gamma_meal, t_lag, peak_mult,
                  time_origin, meal_data, y0, t_span):
    """Solve Bergman ODE system"""

    def ode_func(y, t_val):
        return bergman_ode(y, t_val, p1, p3, beta_meal, gamma_meal,
                           t_lag, peak_mult, time_origin, meal_data)

    soln = odeint(ode_func, y0, t_span)
    G_pred = soln[:, 0] + Gb
    return G_pred


############################################
# 4. LODO with Internal 5-Fold Processing
############################################
def lodo_cv_split(data):
    """
    Leave-One-Day-Out Cross-Validation
    Returns list of (train_dates, test_date) tuples
    """
    sorted_dates = sorted(data['Date'].unique())

    splits = []
    for test_date in sorted_dates:
        train_dates = [d for d in sorted_dates if d != test_date]
        splits.append((train_dates, [test_date]))

    return splits


def split_train_into_5folds(train_data):
    """
    Split training data into 5 folds by dates

    Fold 1: Parameter tuning
    Fold 2-4: Outlier detection
    Fold 5: Internal validation

    Returns:
    --------
    List of 5 DataFrames
    """
    sorted_dates = sorted(train_data['Date'].unique())
    n_days = len(sorted_dates)

    if n_days < 5:
        print(f"  Warning: Only {n_days} training days, cannot split into 5 folds")
        # Return what we can
        folds = []
        for i in range(min(n_days, 5)):
            if i < n_days:
                fold_dates = [sorted_dates[i]]
                fold_data = train_data[train_data['Date'].isin(fold_dates)].copy()
                folds.append(fold_data)
            else:
                folds.append(pd.DataFrame())  # Empty fold
        return folds

    # Split into 5 equal parts
    fold_size = n_days // 5

    folds = []
    for i in range(5):
        start_idx = i * fold_size
        if i == 4:  # Last fold gets remaining days
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
def optimize_parameters(train_data, pbounds, init_points=5, n_iter=15):
    """Optimize model parameters"""
    train_data = train_data.copy()
    train_data.reset_index(drop=True, inplace=True)

    if len(train_data) < 10:
        return None

    time_origin = train_data['Timestamp'].iloc[0]
    train_meals = train_data[
        (train_data['GI'] > 0) & (train_data['Carbs'] > 0)
        ][['Timestamp', 'Meal Type', 'GI', 'Carbs']].copy()

    train_data['t_min'] = (train_data['Timestamp'] - time_origin).dt.total_seconds() / 60
    t_max = train_data['t_min'].max()
    t_span = np.linspace(0, t_max, int(t_max) + 1)

    G0 = train_data['Libre GL'].iloc[0]
    y0 = [max(30, G0 - Gb), 0, 0, 0]

    def objective(p1, p3, beta_meal, gamma_meal, t_lag, peak_mult):
        try:
            G_pred = solve_bergman(
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
# 6. Outlier Detection
############################################
def detect_outliers(fold_data, params, threshold_percentile=95):
    """Detect outlier samples based on prediction error"""
    fold_data = fold_data.copy()
    fold_data.reset_index(drop=True, inplace=True)

    if len(fold_data) < 5:
        return list(fold_data.index), []

    time_origin = fold_data['Timestamp'].iloc[0]
    meals = fold_data[
        (fold_data['GI'] > 0) & (fold_data['Carbs'] > 0)
        ][['Timestamp', 'Meal Type', 'GI', 'Carbs']].copy()

    fold_data['t_min'] = (fold_data['Timestamp'] - time_origin).dt.total_seconds() / 60
    t_max = fold_data['t_min'].max()
    t_span = np.linspace(0, t_max, int(t_max) + 1)

    G0 = fold_data['Libre GL'].iloc[0]
    y0 = [max(30, G0 - Gb), 0, 0, 0]

    try:
        G_pred = solve_bergman(
            params['p1'], params['p3'],
            params['beta_meal'], params['gamma_meal'],
            params['t_lag'], params['peak_mult'],
            time_origin, meals, y0, t_span
        )

        G_pred_interp = np.interp(fold_data['t_min'].values, t_span, G_pred)
        errors = np.abs(fold_data['Libre GL'].values - G_pred_interp)

        threshold = np.percentile(errors, threshold_percentile)

        good_mask = errors <= threshold
        good_indices = fold_data.index[good_mask].tolist()
        outlier_indices = fold_data.index[~good_mask].tolist()

        return good_indices, outlier_indices

    except Exception as e:
        print(f"    Error in outlier detection: {e}")
        return list(fold_data.index), []


############################################
# 7. Last-Step Prediction
############################################
def predict_last_step(data, params, PH_values):
    """Last-step prediction"""
    data = data.copy()
    data.reset_index(drop=True, inplace=True)

    predictions_dict = {PH: [] for PH in PH_values}

    if len(data) < 2:
        return predictions_dict

    time_origin = data['Timestamp'].iloc[0]
    G0 = data['Libre GL'].iloc[0]
    y_current = [max(30, G0 - Gb), 0, 0, 0]

    max_PH = max(PH_values)
    pos = 0
    max_pos = len(data) - 1

    while pos < max_pos:
        try:
            current_time = data.loc[pos, 'Timestamp']
            current_glucose = data.loc[pos, 'Libre GL']

            historical_meals = data[
                (data['Timestamp'] <= current_time) &
                (data['GI'] > 0) &
                (data['Carbs'] > 0)
                ][['Timestamp', 'Meal Type', 'GI', 'Carbs']].copy()

            t_span = np.linspace(0, max_PH, int(max_PH) + 1)

            try:
                G_pred_series = solve_bergman(
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
# 8. Evaluation Metrics
############################################
def evaluate_predictions(predictions_list):
    """Calculate evaluation metrics"""
    if not predictions_list:
        return None

    y_true = np.array([p['y_true'] for p in predictions_list])
    y_pred = np.array([p['y_pred'] for p in predictions_list])

    y_pred = np.clip(y_pred, 40, 400)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    time_in_range = np.mean((y_pred >= 70) & (y_pred <= 180)) * 100
    hyper_events = np.sum(y_pred > 180)
    hypo_events = np.sum(y_pred < 70)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Time_in_Range': time_in_range,
        'Hyper_Events': hyper_events,
        'Hypo_Events': hypo_events,
        'N_Predictions': len(predictions_list)
    }


############################################
# 9. Visualization
############################################
def plot_predictions(predictions, PH, fold_idx, test_date, output_dir):
    """Plot prediction results"""
    if not predictions:
        return

    pred_df = pd.DataFrame(predictions)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(pred_df['target_time'], pred_df['y_true'], 'b-',
             linewidth=2, label='True Glucose', alpha=0.7)
    plt.plot(pred_df['target_time'], pred_df['y_pred'], 'r--',
             linewidth=2, label='Predicted Glucose', alpha=0.7)
    plt.axhline(y=180, color='orange', linestyle='--', alpha=0.5)
    plt.axhline(y=70, color='purple', linestyle='--', alpha=0.5)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Blood Glucose (mg/dL)', fontsize=12)
    plt.title(f'LODO Fold {fold_idx + 1} - PH={PH}min - Test: {test_date}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    plt.scatter(pred_df['y_true'], pred_df['y_pred'], alpha=0.6, s=50)
    min_val = min(pred_df['y_true'].min(), pred_df['y_pred'].min())
    max_val = max(pred_df['y_true'].max(), pred_df['y_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    from sklearn.metrics import r2_score
    r2 = r2_score(pred_df['y_true'], pred_df['y_pred'])
    rmse = np.sqrt(mean_squared_error(pred_df['y_true'], pred_df['y_pred']))

    plt.xlabel('True Glucose (mg/dL)', fontsize=12)
    plt.ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    plt.title(f'R²={r2:.3f}, RMSE={rmse:.2f}', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'lodo_fold{fold_idx + 1}_PH{PH}_test_{test_date}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()


def plot_horizon_comparison(summary_df, output_dir):
    """Plot comparison across horizons"""
    if summary_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].errorbar(summary_df['PH'], summary_df['RMSE_mean'],
                        yerr=summary_df['RMSE_std'], marker='o',
                        linewidth=2, markersize=8, capsize=5)
    axes[0, 0].set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    axes[0, 0].set_ylabel('RMSE (mg/dL)', fontsize=12)
    axes[0, 0].set_title('RMSE vs PH (LODO with Internal 5-Fold)', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].errorbar(summary_df['PH'], summary_df['MAE_mean'],
                        yerr=summary_df['MAE_std'], marker='s',
                        linewidth=2, markersize=8, capsize=5, color='orange')
    axes[0, 1].set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    axes[0, 1].set_ylabel('MAE (mg/dL)', fontsize=12)
    axes[0, 1].set_title('MAE vs PH', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].errorbar(summary_df['PH'], summary_df['MAPE_mean'],
                        yerr=summary_df['MAPE_std'], marker='^',
                        linewidth=2, markersize=8, capsize=5, color='green')
    axes[1, 0].set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    axes[1, 0].set_ylabel('MAPE (%)', fontsize=12)
    axes[1, 0].set_title('MAPE vs PH', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(summary_df['PH'], summary_df['Time_in_Range_mean'],
                   alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Prediction Horizon (minutes)', fontsize=12)
    axes[1, 1].set_ylabel('Time in Range (%)', fontsize=12)
    axes[1, 1].set_title('Time in Range vs PH', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lodo_internal5fold_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


############################################
# 10. Main Function - LODO with Internal 5-Fold
############################################
def main(file_path, prediction_horizons=[15, 30, 45, 60],
         output_dir='results_lodo_internal5fold'):
    """
    LODO with Internal 5-Fold Processing (No Data Leakage)

    For each LODO fold:
        1. Split training data into 5 internal folds
        2. Use Fold 1 for parameter optimization
        3. Use Folds 2-4 for outlier detection
        4. Use Fold 5 for internal validation
        5. Test on the held-out day

    This approach completely avoids data leakage.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print("=" * 80)
    print("LODO WITH INTERNAL 5-FOLD PROCESSING (NO DATA LEAKAGE)")
    print("=" * 80)

    # Load data
    print("\nStep 1: Loading data...")
    try:
        data = load_and_preprocess(file_path)
        data = preprocess_data(data)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    unique_dates = sorted(data['Date'].unique())

    print(f"Total measurements: {len(data)}")
    print(f"Total days: {len(unique_dates)}")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")

    if len(unique_dates) < 3:
        print("Error: Need at least 3 days for LODO")
        return None, None

    # LODO splits
    print("\nStep 2: Creating LODO splits...")
    lodo_splits = lodo_cv_split(data)
    print(f"Number of LODO folds: {len(lodo_splits)}")

    # Parameter bounds
    pbounds = {
        'p1': (0.005, 0.05),
        'p3': (1e-6, 1e-4),
        'beta_meal': (0.01, 0.5),
        'gamma_meal': (0.01, 0.5),
        't_lag': (10, 60),
        'peak_mult': (2.0, 6.0)
    }

    # Store results
    all_results = {PH: [] for PH in prediction_horizons}

    # LODO Loop
    print("\n" + "=" * 80)
    print("Step 3: LODO Cross-Validation with Internal 5-Fold Processing")
    print("=" * 80)

    for lodo_idx, (train_dates, test_dates) in enumerate(lodo_splits):
        test_date = test_dates[0]

        print(f"\n{'=' * 60}")
        print(f"LODO Fold {lodo_idx + 1}/{len(lodo_splits)}")
        print(f"Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
        print(f"Test date: {test_date}")
        print(f"{'=' * 60}")

        # Get training and test data
        train_data = data[data['Date'].isin(train_dates)].copy()
        test_data = data[data['Date'].isin(test_dates)].copy()

        if len(train_data) < 10 or len(test_data) < 5:
            print("Insufficient data, skipping fold")
            continue

        print(f"Train size: {len(train_data)} samples")
        print(f"Test size: {len(test_data)} samples")

        # Internal 5-fold split
        print("\n  Splitting training data into 5 internal folds...")
        internal_folds = split_train_into_5folds(train_data)

        for i, fold in enumerate(internal_folds):
            if len(fold) > 0:
                fold_dates = sorted(fold['Date'].unique())
                print(f"    Internal Fold {i + 1}: {len(fold)} samples, "
                      f"{len(fold_dates)} days ({fold_dates[0]} to {fold_dates[-1]})")
            else:
                print(f"    Internal Fold {i + 1}: Empty")

        # Step A: Parameter optimization on Internal Fold 1
        print("\n  Step A: Optimizing parameters on Internal Fold 1...")
        if len(internal_folds[0]) < 10:
            print("    Insufficient data in Fold 1, skipping this LODO fold")
            continue

        best_params = optimize_parameters(internal_folds[0], pbounds,
                                          init_points=3, n_iter=10)

        if best_params is None:
            print("    Optimization failed, skipping this LODO fold")
            continue

        print("    Optimized parameters:")
        for key, val in best_params.items():
            print(f"      {key:15s}: {val:.6f}")

        # Step B: Outlier detection on Internal Folds 2-4
        print("\n  Step B: Outlier detection on Internal Folds 2-4...")
        total_outliers = 0

        for i in range(1, 4):  # Folds 2, 3, 4 (indices 1, 2, 3)
            if len(internal_folds[i]) < 5:
                print(f"    Internal Fold {i + 1}: Skipping (insufficient data)")
                continue

            print(f"    Internal Fold {i + 1}: Detecting outliers...")
            good_idx, outlier_idx = detect_outliers(internal_folds[i], best_params)

            print(f"      Total: {len(internal_folds[i])}, "
                  f"Good: {len(good_idx)}, Outliers: {len(outlier_idx)}")
            total_outliers += len(outlier_idx)

        print(f"    Total outliers detected: {total_outliers}")

        # Step C: Internal validation on Fold 5
        print("\n  Step C: Internal validation on Fold 5...")
        if len(internal_folds[4]) >= 5:
            val_predictions = predict_last_step(internal_folds[4], best_params,
                                                prediction_horizons)

            for PH in prediction_horizons:
                if val_predictions[PH]:
                    metrics = evaluate_predictions(val_predictions[PH])
                    if metrics:
                        print(f"    PH={PH}min: RMSE={metrics['RMSE']:.2f} mg/dL, "
                              f"N={metrics['N_Predictions']}")
        else:
            print("    Insufficient data for validation")

        # Step D: Test on held-out day
        print(f"\n  Step D: Testing on held-out day ({test_date})...")
        test_predictions = predict_last_step(test_data, best_params,
                                             prediction_horizons)

        for PH in prediction_horizons:
            if test_predictions[PH]:
                metrics = evaluate_predictions(test_predictions[PH])

                if metrics:
                    print(f"    PH={PH}min: RMSE={metrics['RMSE']:.2f} mg/dL, "
                          f"MAE={metrics['MAE']:.2f}, N={metrics['N_Predictions']}")

                    all_results[PH].append({
                        'lodo_fold': lodo_idx + 1,
                        'test_date': test_date,
                        'train_days': len(train_dates),
                        **metrics,
                        **best_params
                    })

                    # Plot (only first 3 folds)
                    if lodo_idx < 3:
                        try:
                            plot_predictions(
                                test_predictions[PH], PH, lodo_idx, test_date,
                                os.path.join(output_dir, 'figures')
                            )
                        except Exception as e:
                            print(f"    Warning: Plot failed - {e}")

    # Aggregate results
    print("\n" + "=" * 80)
    print("FINAL RESULTS - LODO WITH INTERNAL 5-FOLD")
    print("=" * 80)

    summary_results = []
    detailed_results = []

    for PH in prediction_horizons:
        if not all_results[PH]:
            print(f"\nNo results for PH={PH}")
            continue

        results_df = pd.DataFrame(all_results[PH])

        rmse_mean = results_df['RMSE'].mean()
        rmse_std = results_df['RMSE'].std()
        mae_mean = results_df['MAE'].mean()
        mae_std = results_df['MAE'].std()
        mape_mean = results_df['MAPE'].mean()
        mape_std = results_df['MAPE'].std()
        tir_mean = results_df['Time_in_Range'].mean()
        tir_std = results_df['Time_in_Range'].std()

        print(f"\n{'=' * 60}")
        print(f"Prediction Horizon: {PH} minutes")
        print(f"{'=' * 60}")
        print(f"RMSE          : {rmse_mean:7.2f} ± {rmse_std:6.2f} mg/dL")
        print(f"MAE           : {mae_mean:7.2f} ± {mae_std:6.2f} mg/dL")
        print(f"MAPE          : {mape_mean:7.2f} ± {mape_std:6.2f} %")
        print(f"Time in Range : {tir_mean:7.2f} ± {tir_std:6.2f} %")
        print(f"N folds       : {len(results_df)}")

        summary_results.append({
            'PH': PH,
            'RMSE_mean': rmse_mean,
            'RMSE_std': rmse_std,
            'MAE_mean': mae_mean,
            'MAE_std': mae_std,
            'MAPE_mean': mape_mean,
            'MAPE_std': mape_std,
            'Time_in_Range_mean': tir_mean,
            'Time_in_Range_std': tir_std,
            'N_folds': len(results_df)
        })

        results_df.to_csv(
            os.path.join(output_dir, f'lodo_internal5fold_detailed_PH{PH}.csv'),
            index=False
        )

        for _, row in results_df.iterrows():
            detailed_results.append({
                'PH': PH,
                **row.to_dict()
            })

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(os.path.join(output_dir, 'lodo_internal5fold_summary.csv'),
                          index=False)

        if detailed_results:
            all_detailed_df = pd.DataFrame(detailed_results)
            all_detailed_df.to_csv(
                os.path.join(output_dir, 'lodo_internal5fold_all_results.csv'),
                index=False
            )

        plot_horizon_comparison(summary_df, os.path.join(output_dir, 'figures'))

        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print("\nPH (min) | RMSE (mg/dL)    | MAE (mg/dL)     | MAPE (%)        | TIR (%)")
        print("-" * 80)
        for _, row in summary_df.iterrows():
            print(f"{int(row['PH']):8d} | {row['RMSE_mean']:6.2f} ± {row['RMSE_std']:5.2f} | "
                  f"{row['MAE_mean']:6.2f} ± {row['MAE_std']:5.2f} | "
                  f"{row['MAPE_mean']:6.2f} ± {row['MAPE_std']:5.2f} | "
                  f"{row['Time_in_Range_mean']:6.2f} ± {row['Time_in_Range_std']:5.2f}")

        print("\n" + "=" * 80)
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
        print("\nKey Features:")
        print("  ✓ No data leakage (parameters optimized only on training folds)")
        print("  ✓ Internal 5-fold processing for each LODO fold")
        print("  ✓ Systematic outlier detection")
        print("  ✓ Internal validation before testing")
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
    parser = argparse.ArgumentParser(description="Extended Bergman Model runner")
    parser.add_argument("--input_csv", required=True, help="Path to one subject CSV file")
    parser.add_argument("--output_dir", default="results_extended", help="Directory to save outputs")
    parser.add_argument("--prediction_horizons", nargs="+", type=int, default=[15, 30, 45, 60], help="Prediction horizons in minutes")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_df, all_results = main(
        file_path=args.input_csv,
        prediction_horizons=args.prediction_horizons,
        output_dir=args.output_dir,
    )
