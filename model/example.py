import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error

"""
Paper-aligned example-output generator for Scenario 4.

Protocol:
- Outer split: leave-one-day-out (LODO)
- Internal split: first four folds (80%) for parameter selection
- Fifth fold (20%) for internal validation
- Same protocol is applied to both the baseline BMM and the extended BMM

Note:
This script is intended to generate repository example outputs efficiently.
It follows the paper-aligned split logic, but uses a lightweight candidate-search
procedure instead of the full internal optimisation workflow used in the complete
research codebase.
"""

Gb, Ib, n, V1, P2 = 81, 18, 5 / 54, 12, 0.0287

BASELINE_CANDIDATES = [
    [0.022, 9.5e-5, 0.37, 0.30, 18, 1.47],
    [0.050, 1.0e-4, 0.50, 0.50, 34, 1.00],
    [0.031, 1.0e-4, 0.31, 0.50, 14, 1.96],
]

EXTENDED_CANDIDATES = [
    [0.031, 9.8e-5, 0.34, 0.37, 18, 2.42],
    [0.050, 1.0e-6, 0.50, 0.50, 56, 2.00],
    [0.021, 5.5e-5, 0.23, 0.41, 18, 2.77],
]


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Libre GL"]).fillna(0)
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Date"] = df["Timestamp"].dt.date
    df["Libre GL"] = df["Libre GL"].clip(40, 400)
    return df


def split_train_into_5folds(train_data: pd.DataFrame):
    sorted_dates = sorted(train_data["Date"].unique())
    n_days = len(sorted_dates)

    if n_days < 5:
        folds = []
        for i in range(5):
            dates = sorted_dates[i:i + 1] if i < n_days else []
            folds.append(train_data[train_data["Date"].isin(dates)].copy())
        return folds

    fold_size = n_days // 5
    folds = []
    for i in range(5):
        start = i * fold_size
        dates = sorted_dates[start:] if i == 4 else sorted_dates[start:start + fold_size]
        folds.append(train_data[train_data["Date"].isin(dates)].copy())
    return folds


def prepare_day(day_data: pd.DataFrame, extended: bool = False, obs_step: int = 15):
    day_data = day_data.sort_values("Timestamp").reset_index(drop=True)
    start_time = day_data["Timestamp"].iloc[0]

    obs = day_data.iloc[::obs_step].copy().reset_index(drop=True)
    obs_t = ((obs["Timestamp"] - start_time).dt.total_seconds() / 60.0).to_numpy(dtype=float)
    obs_g = obs["Libre GL"].to_numpy(dtype=float)

    if extended:
        meals = day_data[(day_data["Carbs"] > 0) & (day_data["GI"] > 0)][["Timestamp", "Meal Type", "GI", "Carbs"]].copy()
    else:
        meals = day_data[(day_data["Carbs"] > 0)][["Timestamp", "Meal Type", "GI", "Carbs"]].copy()

    meal_t = ((meals["Timestamp"] - start_time).dt.total_seconds() / 60.0).to_numpy(dtype=float)
    carbs = meals["Carbs"].to_numpy(dtype=float) if len(meals) else np.array([], dtype=float)
    gi = meals["GI"].to_numpy(dtype=float) if len(meals) else np.array([], dtype=float)
    meal_types = meals["Meal Type"].astype(str).str.lower().tolist() if len(meals) else []

    if extended and len(meals):
        gi_mult = np.where(
            gi > 70, 1.2 + (gi - 70) * 0.005,
            np.where(gi < 30, 0.8 - (30 - gi) * 0.005, 1.0 + (gi - 50) * 0.004)
        )
        type_mult = np.array([1.2 if "breakfast" in x else 0.9 if "dinner" in x else 1.0 for x in meal_types], dtype=float)
        fg_base = carbs * (gi / 100.0) * gi_mult * type_mult
    else:
        fg_base = carbs

    return {
        "obs_t": obs_t,
        "obs_g": obs_g,
        "meal_t": meal_t,
        "fg_base": fg_base,
        "extended": extended,
    }


def meal_disturbance(t, meal_t, fg_base, beta, gamma, t_lag, peak_mult):
    if len(meal_t) == 0:
        return 0.0
    dt = t - meal_t - t_lag
    mask = dt >= 0
    if not mask.any():
        return 0.0
    d = dt[mask]
    fg = fg_base[mask]
    return float(np.sum(fg * peak_mult * (1.0 - np.exp(-beta * d)) * np.exp(-gamma * d)))


def solve_day(prepared_day, params):
    p1, p3, beta_meal, gamma_meal, t_lag, peak_mult = params
    obs_t = prepared_day["obs_t"]
    obs_g = prepared_day["obs_g"]
    meal_t = prepared_day["meal_t"]
    fg_base = prepared_day["fg_base"]
    t_span = np.arange(0, int(obs_t.max()) + 1, 5.0)

    if prepared_day["extended"]:
        def ode_func(y, t_val):
            G, I, X, E = y
            D = meal_disturbance(t_val, meal_t, fg_base, beta_meal, gamma_meal, t_lag, peak_mult)
            hour_of_day = (t_val / 60.0) % 24.0
            circadian = 1.0 + 0.15 * np.exp(-((hour_of_day - 6.0) ** 2) / 8.0) - 0.1 * np.exp(-((hour_of_day - 18.0) ** 2) / 16.0)
            dEdt = np.clip((G - 15.0), -15.0, 15.0)
            dGdt = (-p1 * G - X * (G + Gb) + D) * circadian
            dIdt = -n * (I + Ib) + n * V1 * (Ib - P2 * p1 * 15.0 / p3 / (15.0 + Gb)) / V1
            dXdt = -P2 * X + p3 * I
            return [dGdt, dIdt, dXdt, dEdt]

        y0 = [max(30.0, obs_g[0] - Gb), 0.0, 0.0, 0.0]
    else:
        def ode_func(y, t_val):
            G, I, X = y
            D = meal_disturbance(t_val, meal_t, fg_base, beta_meal, gamma_meal, t_lag, peak_mult)
            dGdt = -p1 * G - X * (G + Gb) + D
            dIdt = -n * (I + Ib) + n * V1 * (Ib - P2 * p1 * 15.0 / p3 / (15.0 + Gb)) / V1
            dXdt = -P2 * X + p3 * I
            return [dGdt, dIdt, dXdt]

        y0 = [max(30.0, obs_g[0] - Gb), 0.0, 0.0]

    sol = odeint(ode_func, y0, t_span)
    pred = np.interp(obs_t, t_span, sol[:, 0] + Gb)
    return pred


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_random_candidates(base_candidates, n_extra: int, extended: bool, seed: int = 42):
    rng = np.random.default_rng(seed)
    out = [list(x) for x in base_candidates]
    for _ in range(n_extra):
        if extended:
            out.append([
                rng.uniform(0.005, 0.05),
                rng.uniform(1e-6, 1e-4),
                rng.uniform(0.01, 0.5),
                rng.uniform(0.01, 0.5),
                rng.uniform(10, 60),
                rng.uniform(2.0, 6.0),
            ])
        else:
            out.append([
                rng.uniform(0.005, 0.05),
                rng.uniform(1e-6, 1e-4),
                rng.uniform(0.01, 0.5),
                rng.uniform(0.01, 0.5),
                rng.uniform(10, 60),
                rng.uniform(1.0, 4.0),
            ])
    return out


def choose_params(calib_days, valid_days, candidates):
    best_params = None
    best_score = np.inf

    for params in candidates:
        calib_rmse = np.mean([rmse(day["obs_g"], solve_day(day, params)) for day in calib_days])
        valid_rmse = np.mean([rmse(day["obs_g"], solve_day(day, params)) for day in valid_days])
        score = 0.3 * calib_rmse + 0.7 * valid_rmse
        if score < best_score:
            best_score = score
            best_params = params

    return np.array(best_params, dtype=float)


def evaluate_test_day(day_data, params, extended: bool, prediction_horizons, obs_step: int = 10):
    prepared = prepare_day(day_data, extended=extended, obs_step=obs_step)
    pred = solve_day(prepared, params)
    obs_t = prepared["obs_t"]
    obs_g = prepared["obs_g"]

    out = {}
    for ph in prediction_horizons:
        y_true, y_pred = [], []
        for t_val in obs_t:
            target_t = t_val + ph
            if target_t > obs_t.max():
                continue
            y_true.append(np.interp(target_t, obs_t, obs_g))
            y_pred.append(np.interp(target_t, obs_t, pred))
        out[ph] = rmse(y_true, y_pred)
    return out


def run_example(input_csv: str, output_dir: str, prediction_horizons, n_lodo_folds: int, n_extra_candidates: int):
    df = load_data(input_csv)
    all_dates = sorted(df["Date"].unique())[:n_lodo_folds]

    baseline_candidates = add_random_candidates(BASELINE_CANDIDATES, n_extra_candidates, extended=False, seed=42)
    extended_candidates = add_random_candidates(EXTENDED_CANDIDATES, n_extra_candidates, extended=True, seed=42)

    baseline_rows = []
    extended_rows = []

    for test_date in all_dates:
        train_data = df[df["Date"] != test_date].copy()
        test_data = df[df["Date"] == test_date].copy()
        folds = split_train_into_5folds(train_data)

        calib_data = pd.concat([f for f in folds[:4] if len(f) > 0], ignore_index=True)
        valid_data = folds[4].copy()

        calib_days_baseline = [prepare_day(calib_data[calib_data["Date"] == d], extended=False, obs_step=20) for d in sorted(calib_data["Date"].unique())]
        calib_days_extended = [prepare_day(calib_data[calib_data["Date"] == d], extended=True, obs_step=20) for d in sorted(calib_data["Date"].unique())]
        valid_days_baseline = [prepare_day(valid_data[valid_data["Date"] == d], extended=False, obs_step=20) for d in sorted(valid_data["Date"].unique())]
        valid_days_extended = [prepare_day(valid_data[valid_data["Date"] == d], extended=True, obs_step=20) for d in sorted(valid_data["Date"].unique())]

        baseline_params = choose_params(calib_days_baseline, valid_days_baseline, baseline_candidates)
        extended_params = choose_params(calib_days_extended, valid_days_extended, extended_candidates)

        baseline_metrics = evaluate_test_day(test_data, baseline_params, extended=False, prediction_horizons=prediction_horizons, obs_step=10)
        extended_metrics = evaluate_test_day(test_data, extended_params, extended=True, prediction_horizons=prediction_horizons, obs_step=10)

        for ph in prediction_horizons:
            baseline_rows.append({"Test Day": str(test_date), "PH": ph, "RMSE": baseline_metrics[ph]})
            extended_rows.append({"Test Day": str(test_date), "PH": ph, "RMSE": extended_metrics[ph]})

    baseline_df = pd.DataFrame(baseline_rows)
    extended_df = pd.DataFrame(extended_rows)

    baseline_summary = baseline_df.groupby("PH")["RMSE"].agg(["mean", "std", "count"]).reset_index()
    baseline_summary.columns = ["PH", "RMSE_mean", "RMSE_std", "N"]
    extended_summary = extended_df.groupby("PH")["RMSE"].agg(["mean", "std", "count"]).reset_index()
    extended_summary.columns = ["PH", "RMSE_mean", "RMSE_std", "N"]

    comparison = baseline_summary[["PH", "RMSE_mean"]].merge(
        extended_summary[["PH", "RMSE_mean"]], on="PH", suffixes=("_baseline", "_extended")
    )
    comparison["Improvement"] = comparison["RMSE_mean_baseline"] - comparison["RMSE_mean_extended"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary.to_csv(out_dir / "baseline_003_summary_paper_aligned.csv", index=False)
    extended_summary.to_csv(out_dir / "extended_003_summary_paper_aligned.csv", index=False)
    comparison.to_csv(out_dir / "comparison_003_summary_paper_aligned.csv", index=False)

    print("Saved example outputs to:", out_dir)
    print("\nBaseline summary:\n", baseline_summary.to_string(index=False))
    print("\nExtended summary:\n", extended_summary.to_string(index=False))
    print("\nComparison summary:\n", comparison.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="example_outputs_paper_aligned")
    parser.add_argument("--n_lodo_folds", type=int, default=3)
    parser.add_argument("--n_extra_candidates", type=int, default=8)
    parser.add_argument("--prediction_horizons", type=int, nargs="+", default=[15, 30, 45, 60])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_example(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        prediction_horizons=args.prediction_horizons,
        n_lodo_folds=args.n_lodo_folds,
        n_extra_candidates=args.n_extra_candidates,
    )
