import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error


@dataclass
class Config:
    Gb: float = 81.0
    Ib: float = 14.0
    n: float = 5.0 / 54.0
    p2: float = 0.0287
    h: float = 80.0               # pancreatic secretion threshold (mg/dL)
    gamma: float = 0.01           # pancreatic secretion gain
    dt_minutes: int = 1
    day_minutes: int = 1440
    random_state: int = 42
    maxiter: int = 12             # differential evolution iterations (minimal release)
    popsize: int = 8              # differential evolution population size


REQUIRED_COLUMNS = ["Timestamp", "Libre GL", "Meal Type", "GI", "Carbs"]


# -----------------------------
# Data utilities
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Libre GL"]).copy()
    df = df.sort_values("Timestamp").reset_index(drop=True)

    df["Libre GL"] = pd.to_numeric(df["Libre GL"], errors="coerce")
    df = df.dropna(subset=["Libre GL"]).copy()

    # Minimal preprocessing: small-gap time interpolation for glucose only.
    df = df.set_index("Timestamp")
    df["Libre GL"] = df["Libre GL"].interpolate(method="time", limit_direction="both")
    df = df.reset_index()

    # Event channels
    df["Carbs"] = pd.to_numeric(df["Carbs"], errors="coerce").fillna(0.0)
    df["GI"] = pd.to_numeric(df["GI"], errors="coerce").fillna(0.0)
    df["Meal Type"] = df["Meal Type"].fillna("unknown").astype(str)

    df["Libre GL"] = df["Libre GL"].clip(40.0, 400.0)
    df["Date"] = df["Timestamp"].dt.date
    return df


def add_relative_time(df: pd.DataFrame, ref_time: pd.Timestamp = None) -> pd.DataFrame:
    df = df.copy().sort_values("Timestamp").reset_index(drop=True)
    start_time = df["Timestamp"].min() if ref_time is None else ref_time
    df["t_min"] = (df["Timestamp"] - start_time).dt.total_seconds() / 60.0
    return df


def lodo_splits(df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, object]]:
    splits: List[Tuple[pd.DataFrame, pd.DataFrame, object]] = []
    for test_day in sorted(df["Date"].unique()):
        train_df = df[df["Date"] != test_day].copy()
        test_df = df[df["Date"] == test_day].copy()
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        splits.append((train_df, test_df, test_day))
    return splits


def chronological_train_valid_split_by_day(
    train_df: pd.DataFrame, ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.sort_values("Timestamp").reset_index(drop=True)
    unique_dates = sorted(train_df["Date"].unique())
    if len(unique_dates) == 1:
        return train_df.copy(), train_df.copy()

    split_idx = max(1, int(np.floor(len(unique_dates) * ratio)))
    if split_idx >= len(unique_dates):
        split_idx = len(unique_dates) - 1

    calib_dates = unique_dates[:split_idx]
    valid_dates = unique_dates[split_idx:]

    calib_df = train_df[train_df["Date"].isin(calib_dates)].copy()
    valid_df = train_df[train_df["Date"].isin(valid_dates)].copy()
    return calib_df, valid_df


# -----------------------------
# Meal-effect utilities
# -----------------------------
def phi_gi(gi_value: float) -> float:
    if gi_value > 70:
        return 1.2
    if gi_value < 30:
        return 0.8
    return 1.0



def phi_type(meal_type: str) -> float:
    meal_type = str(meal_type).strip().lower()
    if "breakfast" in meal_type:
        return 1.05
    if "lunch" in meal_type:
        return 1.00
    if "dinner" in meal_type:
        return 0.95
    if "snack" in meal_type:
        return 0.90
    return 1.00



def absorption_kernel(
    dt: float,
    beta_meal: float,
    lambda_decay: float,
    t_lag: float,
    peak_mult: float,
) -> float:
    shifted_dt = dt - t_lag
    if shifted_dt < 0:
        return 0.0
    return peak_mult * (1.0 - np.exp(-beta_meal * shifted_dt)) * np.exp(-lambda_decay * shifted_dt)



def extract_meals(df: pd.DataFrame) -> pd.DataFrame:
    meal_df = df[["Timestamp", "Meal Type", "GI", "Carbs"]].copy()
    meal_df = meal_df[(meal_df["GI"] > 0) & (meal_df["Carbs"] > 0)].reset_index(drop=True)
    return meal_df



def meal_disturbance_extended(
    t_val: float,
    meal_data: pd.DataFrame,
    current_start: pd.Timestamp,
    beta_meal: float,
    lambda_decay: float,
    t_lag: float,
    peak_mult: float,
) -> float:
    total = 0.0
    if meal_data.empty:
        return total

    for _, row in meal_data.iterrows():
        meal_t = (row["Timestamp"] - current_start).total_seconds() / 60.0
        dt = t_val - meal_t
        fm = (
            float(row["Carbs"])
            * float(row["GI"]) / 100.0
            * phi_gi(float(row["GI"]))
            * phi_type(row["Meal Type"])
        )
        total += fm * absorption_kernel(dt, beta_meal, lambda_decay, t_lag, peak_mult)
    return total



def meal_disturbance_baseline(
    t_val: float,
    meal_data: pd.DataFrame,
    current_start: pd.Timestamp,
    beta_meal: float,
    lambda_decay: float,
    t_lag: float,
    peak_mult: float,
) -> float:
    total = 0.0
    if meal_data.empty:
        return total

    for _, row in meal_data.iterrows():
        meal_t = (row["Timestamp"] - current_start).total_seconds() / 60.0
        dt = t_val - meal_t
        fm = float(row["Carbs"])
        total += fm * absorption_kernel(dt, beta_meal, lambda_decay, t_lag, peak_mult)
    return total



def circadian_factor(t_val: float) -> float:
    hour_of_day = (t_val / 60.0) % 24.0
    return 1.0 + 0.1 * np.sin(2.0 * np.pi * (hour_of_day - 6.0) / 24.0)


# -----------------------------
# ODE systems
# -----------------------------
def bergman_original(
    y: List[float],
    t_val: float,
    params: Dict[str, float],
    cfg: Config,
    u_t: float = 0.0,
) -> List[float]:
    """Classical Bergman Minimal Model (baseline physiological model)."""
    G, X, I = y
    secretion = cfg.gamma * max(0.0, G - cfg.h)
    dGdt = -params["p1"] * (G - cfg.Gb) - G * X
    dXdt = -cfg.p2 * X + params["p3"] * (I - cfg.Ib)
    dIdt = -cfg.n * (I - cfg.Ib) + secretion + u_t
    return [dGdt, dXdt, dIdt]



def bergman_baseline_meal(
    y: List[float],
    t_val: float,
    params: Dict[str, float],
    meal_data: pd.DataFrame,
    current_start: pd.Timestamp,
    cfg: Config,
) -> List[float]:
    G, X, I = y
    d_meal = meal_disturbance_baseline(
        t_val=t_val,
        meal_data=meal_data,
        current_start=current_start,
        beta_meal=params["beta_meal"],
        lambda_decay=params["lambda_decay"],
        t_lag=params["t_lag"],
        peak_mult=params["peak_mult"],
    )
    secretion = cfg.gamma * max(0.0, G - cfg.h)
    dGdt = -params["p1"] * (G - cfg.Gb) - G * X + d_meal
    dXdt = -cfg.p2 * X + params["p3"] * (I - cfg.Ib)
    dIdt = -cfg.n * (I - cfg.Ib) + secretion
    return [dGdt, dXdt, dIdt]



def bergman_extended(
    y: List[float],
    t_val: float,
    params: Dict[str, float],
    meal_data: pd.DataFrame,
    current_start: pd.Timestamp,
    cfg: Config,
) -> List[float]:
    G, X, I = y
    d_meal = meal_disturbance_extended(
        t_val=t_val,
        meal_data=meal_data,
        current_start=current_start,
        beta_meal=params["beta_meal"],
        lambda_decay=params["lambda_decay"],
        t_lag=params["t_lag"],
        peak_mult=params["peak_mult"],
    )
    secretion = cfg.gamma * max(0.0, G - cfg.h)
    dGdt = -params["p1"] * (G - cfg.Gb) - G * X + d_meal * circadian_factor(t_val)
    dXdt = -cfg.p2 * X + params["p3"] * (I - cfg.Ib)
    dIdt = -cfg.n * (I - cfg.Ib) + secretion
    return [dGdt, dXdt, dIdt]



def solve_original_bmm(params: Dict[str, float], cfg: Config, t_end: float) -> Tuple[np.ndarray, np.ndarray]:
    t_grid = np.arange(0.0, t_end + cfg.dt_minutes, cfg.dt_minutes)
    y0 = [cfg.Gb, 0.0, cfg.Ib]

    def ode_func(y, t_val):
        return bergman_original(y, t_val, params, cfg)

    sol = odeint(ode_func, y0, t_grid)
    return t_grid, sol[:, 0]



def solve_meal_model(
    params: Dict[str, float],
    meal_data: pd.DataFrame,
    current_start: pd.Timestamp,
    cfg: Config,
    t_end: float,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    t_grid = np.arange(0.0, t_end + cfg.dt_minutes, cfg.dt_minutes)
    y0 = [cfg.Gb, 0.0, cfg.Ib]

    if model_name == "baseline_meal":
        def ode_func(y, t_val):
            return bergman_baseline_meal(y, t_val, params, meal_data, current_start, cfg)
    elif model_name == "extended":
        def ode_func(y, t_val):
            return bergman_extended(y, t_val, params, meal_data, current_start, cfg)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    sol = odeint(ode_func, y0, t_grid)
    return t_grid, sol[:, 0]


# -----------------------------
# Evaluation utilities
# -----------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))



def one_day_fit_rmse(day_df: pd.DataFrame, params: Dict[str, float], cfg: Config, model_name: str) -> float:
    day_df = add_relative_time(day_df)
    day_start = day_df["Timestamp"].min()
    obs_t = day_df["t_min"].to_numpy(dtype=float)
    obs_g = day_df["Libre GL"].to_numpy(dtype=float)

    if model_name == "original_bmm":
        t_grid, g_pred = solve_original_bmm(params, cfg, t_end=float(obs_t.max()))
    else:
        meals = extract_meals(day_df)
        t_grid, g_pred = solve_meal_model(params, meals, day_start, cfg, t_end=float(obs_t.max()), model_name=model_name)

    g_at_obs = np.interp(obs_t, t_grid, g_pred)
    return rmse(obs_g, g_at_obs)



def evaluate_days_fit(days_df: pd.DataFrame, params: Dict[str, float], cfg: Config, model_name: str) -> float:
    rmses = []
    for day in sorted(days_df["Date"].unique()):
        day_df = days_df[days_df["Date"] == day].copy()
        if len(day_df) < 2:
            continue
        rmses.append(one_day_fit_rmse(day_df, params, cfg, model_name))
    return float(np.mean(rmses)) if rmses else np.nan



def predict_horizon_no_future_leakage(
    model_params: Dict[str, float],
    test_df: pd.DataFrame,
    cfg: Config,
    prediction_horizon: int,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    test_df = add_relative_time(test_df)
    test_start = test_df["Timestamp"].min()
    full_meals = extract_meals(test_df)
    obs_t = test_df["t_min"].to_numpy(dtype=float)
    obs_g = test_df["Libre GL"].to_numpy(dtype=float)
    obs_ts = test_df["Timestamp"].to_numpy()

    y_true: List[float] = []
    y_pred: List[float] = []

    for i, current_t in enumerate(obs_t):
        target_t = current_t + prediction_horizon
        if target_t > float(obs_t.max()):
            continue

        current_ts = pd.Timestamp(obs_ts[i])
        visible_meals = full_meals[full_meals["Timestamp"] <= current_ts].copy()

        if model_name == "original_bmm":
            t_grid, g_pred = solve_original_bmm(model_params, cfg, t_end=target_t)
        else:
            t_grid, g_pred = solve_meal_model(
                model_params,
                visible_meals,
                test_start,
                cfg,
                t_end=target_t,
                model_name=model_name,
            )

        y_pred.append(float(np.interp(target_t, t_grid, g_pred)))
        y_true.append(float(np.interp(target_t, obs_t, obs_g)))

    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


# -----------------------------
# Optimisation
# -----------------------------
def optimize_original_bmm_parameters(calib_df: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    bounds = [
        (0.001, 0.05),    # p1
        (1e-6, 1e-4),     # p3
    ]

    def objective(theta):
        p1, p3 = theta
        params = {"p1": float(p1), "p3": float(p3)}
        return evaluate_days_fit(calib_df, params, cfg, model_name="original_bmm")

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=cfg.random_state,
        maxiter=cfg.maxiter,
        popsize=cfg.popsize,
        polish=False,
        updating="deferred",
    )
    p1, p3 = result.x
    return {"p1": float(p1), "p3": float(p3)}



def optimize_meal_model_parameters(calib_df: pd.DataFrame, cfg: Config, model_name: str) -> Dict[str, float]:
    bounds = [
        (0.001, 0.05),   # p1
        (1e-6, 1e-4),    # p3
        (0.005, 1.0),    # beta_meal
        (0.005, 1.0),    # lambda_decay
        (0.0, 90.0),     # t_lag
        (1.0, 6.0),      # peak_mult
    ]

    def objective(theta):
        p1, p3, beta_meal, lambda_decay, t_lag, peak_mult = theta
        params = {
            "p1": float(p1),
            "p3": float(p3),
            "beta_meal": float(beta_meal),
            "lambda_decay": float(lambda_decay),
            "t_lag": float(t_lag),
            "peak_mult": float(peak_mult),
        }
        return evaluate_days_fit(calib_df, params, cfg, model_name=model_name)

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=cfg.random_state,
        maxiter=cfg.maxiter,
        popsize=cfg.popsize,
        polish=False,
        updating="deferred",
    )
    p1, p3, beta_meal, lambda_decay, t_lag, peak_mult = result.x
    return {
        "p1": float(p1),
        "p3": float(p3),
        "beta_meal": float(beta_meal),
        "lambda_decay": float(lambda_decay),
        "t_lag": float(t_lag),
        "peak_mult": float(peak_mult),
    }


# -----------------------------
# Main experiment
# -----------------------------
def run_scenario4(input_csv: str, output_dir: str, prediction_horizons: List[int]) -> pd.DataFrame:
    cfg = Config()
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(input_csv)
    splits = lodo_splits(df)
    all_results = []

    for train_df, test_df, test_day in splits:
        calib_df, valid_df = chronological_train_valid_split_by_day(train_df, ratio=0.8)

        params_original = optimize_original_bmm_parameters(calib_df, cfg)
        params_baseline_meal = optimize_meal_model_parameters(calib_df, cfg, model_name="baseline_meal")
        params_extended = optimize_meal_model_parameters(calib_df, cfg, model_name="extended")

        calib_rmse_original = evaluate_days_fit(calib_df, params_original, cfg, model_name="original_bmm")
        valid_rmse_original = evaluate_days_fit(valid_df, params_original, cfg, model_name="original_bmm")

        calib_rmse_baseline_meal = evaluate_days_fit(calib_df, params_baseline_meal, cfg, model_name="baseline_meal")
        valid_rmse_baseline_meal = evaluate_days_fit(valid_df, params_baseline_meal, cfg, model_name="baseline_meal")

        calib_rmse_extended = evaluate_days_fit(calib_df, params_extended, cfg, model_name="extended")
        valid_rmse_extended = evaluate_days_fit(valid_df, params_extended, cfg, model_name="extended")

        for ph in prediction_horizons:
            y_true_ext, y_pred_ext = predict_horizon_no_future_leakage(
                params_extended, test_df, cfg, prediction_horizon=ph, model_name="extended"
            )
            y_true_base_meal, y_pred_base_meal = predict_horizon_no_future_leakage(
                params_baseline_meal, test_df, cfg, prediction_horizon=ph, model_name="baseline_meal"
            )
            y_true_orig, y_pred_orig = predict_horizon_no_future_leakage(
                params_original, test_df, cfg, prediction_horizon=ph, model_name="original_bmm"
            )

            all_results.append({
                "Test Day": str(test_day),
                "PH": ph,
                "RMSE_Extended": rmse(y_true_ext, y_pred_ext),
                "RMSE_BaselineMeal": rmse(y_true_base_meal, y_pred_base_meal),
                "RMSE_OriginalBMM": rmse(y_true_orig, y_pred_orig),
                "CalibRMSE_Extended": calib_rmse_extended,
                "ValidRMSE_Extended": valid_rmse_extended,
                "CalibRMSE_BaselineMeal": calib_rmse_baseline_meal,
                "ValidRMSE_BaselineMeal": valid_rmse_baseline_meal,
                "CalibRMSE_OriginalBMM": calib_rmse_original,
                "ValidRMSE_OriginalBMM": valid_rmse_original,
                "Extended_p1": params_extended["p1"],
                "Extended_p3": params_extended["p3"],
                "Extended_beta_meal": params_extended["beta_meal"],
                "Extended_lambda_decay": params_extended["lambda_decay"],
                "Extended_t_lag": params_extended["t_lag"],
                "Extended_peak_mult": params_extended["peak_mult"],
                "BaselineMeal_p1": params_baseline_meal["p1"],
                "BaselineMeal_p3": params_baseline_meal["p3"],
                "BaselineMeal_beta_meal": params_baseline_meal["beta_meal"],
                "BaselineMeal_lambda_decay": params_baseline_meal["lambda_decay"],
                "BaselineMeal_t_lag": params_baseline_meal["t_lag"],
                "BaselineMeal_peak_mult": params_baseline_meal["peak_mult"],
                "OriginalBMM_p1": params_original["p1"],
                "OriginalBMM_p3": params_original["p3"],
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "scenario4_detailed_results.csv"), index=False)

    summary_df = (
        results_df.groupby("PH")[["RMSE_Extended", "RMSE_BaselineMeal", "RMSE_OriginalBMM"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_df.columns = [
        "PH",
        "Extended_mean", "Extended_std", "Extended_N",
        "BaselineMeal_mean", "BaselineMeal_std", "BaselineMeal_N",
        "OriginalBMM_mean", "OriginalBMM_std", "OriginalBMM_N",
    ]
    summary_df.to_csv(os.path.join(output_dir, "scenario4_summary_results.csv"), index=False)

    comparison_df = summary_df[["PH", "BaselineMeal_mean", "Extended_mean", "OriginalBMM_mean"]].copy()
    comparison_df["Improvement_vs_BaselineMeal"] = comparison_df["BaselineMeal_mean"] - comparison_df["Extended_mean"]
    comparison_df["Improvement_vs_OriginalBMM"] = comparison_df["OriginalBMM_mean"] - comparison_df["Extended_mean"]
    comparison_df.to_csv(os.path.join(output_dir, "scenario4_comparison_results.csv"), index=False)

    return results_df


# -----------------------------
# Demo data
# -----------------------------
def generate_demo_subject(output_csv: str, n_days: int = 6, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01 00:00:00")
    rows = []

    for day in range(n_days):
        day_start = start + pd.Timedelta(days=day)
        times = pd.date_range(day_start, periods=288, freq="5min")
        base = 110 + 12 * np.sin(np.linspace(0, 2 * np.pi, len(times)))
        glucose = base + rng.normal(0, 5, len(times))

        meal_slots = {
            8 * 60: ("Breakfast", 45, 55),
            13 * 60: ("Lunch", 65, 68),
            19 * 60: ("Dinner", 80, 78),
        }

        for i, ts in enumerate(times):
            minute = ts.hour * 60 + ts.minute
            meal_type, carbs, gi = meal_slots.get(minute, (np.nan, 0, 0))
            rows.append({
                "Timestamp": ts,
                "Libre GL": float(glucose[i]),
                "Meal Type": meal_type,
                "GI": gi,
                "Carbs": carbs,
            })

    pd.DataFrame(rows).to_csv(output_csv, index=False)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal paper-aligned Scenario 4 implementation with fair baseline comparison."
    )
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="scenario4_outputs")
    parser.add_argument("--prediction_horizons", type=int, nargs="+", default=[15, 30, 45, 60])
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()



def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_csv = args.input_csv
    if args.demo:
        input_csv = os.path.join(args.output_dir, "demo_subject.csv")
        generate_demo_subject(input_csv)

    if input_csv is None:
        raise ValueError("Provide --input_csv or use --demo.")

    results_df = run_scenario4(
        input_csv=input_csv,
        output_dir=args.output_dir,
        prediction_horizons=args.prediction_horizons,
    )

    print("\nDetailed results preview:")
    print(results_df.head())
    print(f"\nSaved summary to: {os.path.join(args.output_dir, 'scenario4_summary_results.csv')}")


if __name__ == "__main__":
    main()
