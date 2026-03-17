"""
Data Preprocessing Pipeline
============================
Implements Section 2(b) of the paper:
  - CGM signal denoising via median filter
  - Gap handling (interpolate <30min, exclude >=30min)
  - Temporal alignment to CGM grid
  - Meal log alignment and encoding

Designed to work with the PhysioNet CGMacros Dataset (v1.0.0).
Download from: https://doi.org/10.13026/3z8q-x658
"""

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from models.glucose_model import MealEvent, GIEstimator


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

CGM_INTERVAL_MIN   = 5      # minutes between CGM samples
INTERP_GAP_MAX_MIN = 30     # gaps ≤ this are interpolated
MEDIAN_FILTER_KERNEL = 5    # window size for median denoising (samples)
PREPRANDIAL_WINDOW = 30     # minutes before meal for baseline glucose

# CGMacros column names (adapt if your CSV differs)
COL_CGM_TIME    = "time"
COL_CGM_GLUCOSE = "glucose"
COL_MEAL_TIME   = "meal_time"
COL_MEAL_TYPE   = "meal_type"
COL_CARBS       = "carbs_g"
COL_FIBER       = "fiber_g"
COL_PROTEIN     = "protein_g"
COL_FAT         = "fat_g"


# ─────────────────────────────────────────────
# CGM preprocessing
# ─────────────────────────────────────────────

def load_cgm_csv(filepath: str) -> pd.DataFrame:
    """Load CGM CSV and parse timestamps."""
    df = pd.read_csv(filepath, parse_dates=[COL_CGM_TIME])
    df = df.sort_values(COL_CGM_TIME).reset_index(drop=True)
    return df


def cgm_to_minutes(df: pd.DataFrame,
                   reference_time: Optional[pd.Timestamp] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Convert CGM timestamps to elapsed minutes from reference."""
    if reference_time is None:
        reference_time = df[COL_CGM_TIME].iloc[0]
    times_min = (df[COL_CGM_TIME] - reference_time).dt.total_seconds() / 60.0
    glucose   = df[COL_CGM_GLUCOSE].values.astype(float)
    return times_min.values, glucose


def detect_gaps(times: np.ndarray,
                max_gap: float = INTERP_GAP_MAX_MIN) -> np.ndarray:
    """
    Return boolean mask: True where sample follows a gap > max_gap.
    Used to exclude training windows spanning large gaps.
    """
    diffs = np.diff(times, prepend=times[0])
    return diffs > max_gap


def interpolate_cgm(times: np.ndarray,
                    glucose: np.ndarray,
                    target_interval: float = CGM_INTERVAL_MIN,
                    max_gap: float = INTERP_GAP_MAX_MIN) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build uniform 5-min time grid and fill values:
      - gaps ≤ max_gap  →  linear interpolation
      - gaps >  max_gap →  NaN (excluded from training)
    """
    t_start = times[0]
    t_end   = times[-1]
    t_grid  = np.arange(t_start, t_end + target_interval, target_interval)

    # Linear interpolation on full grid
    g_interp = np.interp(t_grid, times, glucose)

    # Mask out segments following large gaps
    gap_mask = detect_gaps(times, max_gap)
    gap_starts = times[gap_mask]
    gap_ends   = times[gap_mask] - np.diff(times, prepend=times[0])[gap_mask]

    # For each large gap, NaN out the interpolated region
    orig_gaps = np.where(np.diff(times) > max_gap)[0]
    for gi in orig_gaps:
        t_gap_end   = times[gi + 1]
        t_gap_start = times[gi]
        nan_mask = (t_grid > t_gap_start) & (t_grid < t_gap_end)
        g_interp[nan_mask] = np.nan

    return t_grid, g_interp


def denoise_cgm(glucose: np.ndarray,
                kernel_size: int = MEDIAN_FILTER_KERNEL) -> np.ndarray:
    """Apply median filter for denoising. Section 2(b)."""
    # Handle NaN by forward-fill before filtering, then restore NaN
    nan_mask = np.isnan(glucose)
    g_filled = glucose.copy()
    # simple forward-fill
    for i in range(1, len(g_filled)):
        if np.isnan(g_filled[i]):
            g_filled[i] = g_filled[i - 1]
    g_filtered = medfilt(g_filled, kernel_size=kernel_size)
    g_filtered[nan_mask] = np.nan
    return g_filtered


def preprocess_cgm(filepath: str,
                   reference_time: Optional[pd.Timestamp] = None
                   ) -> Dict[str, np.ndarray]:
    """
    Full CGM preprocessing pipeline. Returns dict with keys:
      t_raw, g_raw      – original data in minutes
      t_grid, g_grid    – uniform 5-min grid, gaps NaN'd
      g_denoised        – median-filtered grid signal
      day_labels        – integer day index per grid point
    """
    df = load_cgm_csv(filepath)
    t_raw, g_raw = cgm_to_minutes(df, reference_time)

    t_grid, g_grid = interpolate_cgm(t_raw, g_raw)
    g_denoised = denoise_cgm(g_grid)

    # Day labels (integer day from recording start)
    day_labels = (t_grid // (24 * 60)).astype(int)

    return {
        "t_raw":      t_raw,
        "g_raw":      g_raw,
        "t_grid":     t_grid,
        "g_grid":     g_grid,
        "g_denoised": g_denoised,
        "day_labels": day_labels,
    }


# ─────────────────────────────────────────────
# Meal log preprocessing
# ─────────────────────────────────────────────

def load_meal_csv(filepath: str,
                  reference_time: Optional[pd.Timestamp] = None) -> List[MealEvent]:
    """
    Load meal log CSV and return list of MealEvent objects.
    Missing macronutrient values are zero-imputed (non-carb fields).
    Meal type is preserved as 'unknown' if missing (not zero-imputed).
    """
    df = pd.read_csv(filepath, parse_dates=[COL_MEAL_TIME])
    df = df.sort_values(COL_MEAL_TIME).reset_index(drop=True)

    if reference_time is None:
        reference_time = df[COL_MEAL_TIME].iloc[0].normalize()

    meals = []
    for _, row in df.iterrows():
        elapsed = (row[COL_MEAL_TIME] - reference_time).total_seconds() / 60.0

        # Zero-impute numeric macro fields (not NaN-preserving – Section 2b)
        carbs   = float(row.get(COL_CARBS, 0)   or 0)
        fiber   = float(row.get(COL_FIBER, 0)   or 0)
        protein = float(row.get(COL_PROTEIN, 0) or 0)
        fat     = float(row.get(COL_FAT, 0)     or 0)

        # Preserve meal type as unknown if missing
        m_type = str(row.get(COL_MEAL_TYPE, "unknown") or "unknown").lower()

        meals.append(MealEvent(
            time_min  = elapsed,
            carbs_g   = carbs,
            fiber_g   = fiber,
            protein_g = protein,
            fat_g     = fat,
            meal_type = m_type,
        ))
    return meals


def estimate_reference_iauc(cgm_data: Dict[str, np.ndarray],
                             reference_meal_idx: int = 0) -> float:
    """
    Estimate iAUC of a reference food (white bread) using first meal.
    In practice, this should be computed from a standardised test if available.
    Fallback: use the first recorded meal's postprandial response.
    """
    est = GIEstimator()
    t   = cgm_data["t_grid"]
    g   = cgm_data["g_denoised"]
    # Use mean postprandial response as reference proxy
    meal_times = t[::60][:3]   # approximate 3 reference times
    iaucs = [est.iauc(g, t, mt) for mt in meal_times if mt + 180 <= t[-1]]
    if iaucs:
        return float(np.mean(iaucs))
    return 100.0   # safe default


def prepare_subject_data(cgm_filepath: str,
                         meal_filepath: str,
                         estimate_gi: bool = True) -> dict:
    """
    Full subject data preparation pipeline.
    Returns a dict ready for `fit_subject_lodocv`.
    """
    cgm_data = preprocess_cgm(cgm_filepath)
    meals    = load_meal_csv(meal_filepath)

    if estimate_gi:
        ref_iauc = estimate_reference_iauc(cgm_data)
        estimator = GIEstimator()
        meals = estimator.batch_estimate(
            meals,
            cgm_data["g_denoised"],
            cgm_data["t_grid"],
            ref_iauc
        )

    # Remove NaN grid points from training data
    valid_mask = ~np.isnan(cgm_data["g_denoised"])

    return {
        "times":       cgm_data["t_grid"][valid_mask],
        "glucose":     cgm_data["g_denoised"][valid_mask],
        "days":        cgm_data["day_labels"][valid_mask],
        "meals":       meals,
        "cgm_full":    cgm_data,
    }


# ─────────────────────────────────────────────
# Synthetic data generator (for testing / demo)
# ─────────────────────────────────────────────

def generate_synthetic_subject(n_days: int = 5,
                                n_meals_per_day: int = 3,
                                noise_std: float = 5.0,
                                seed: int = 0) -> dict:
    """
    Generate synthetic CGM + meal data for one subject.
    Used for unit testing and demo runs without real data.
    """
    rng = np.random.RandomState(seed)
    from models.glucose_model import SubjectParameters, GlucoseModel

    params = SubjectParameters(
        p1=0.015, p3=5e-5, beta=0.08, lam=0.06, tau=25.0, peak=3.0
    )

    # Fixed meal schedule per day
    base_meal_times_h  = [8.0, 13.0, 19.0]   # breakfast, lunch, dinner
    base_meal_types    = ["breakfast", "lunch", "dinner"]
    base_carbs         = [60.0, 80.0, 70.0]
    base_gi            = [65.0, 55.0, 50.0]

    all_meals: List[MealEvent] = []
    for d in range(n_days):
        for mt, mtype, carbs, gi in zip(base_meal_times_h,
                                         base_meal_types,
                                         base_carbs, base_gi):
            t_m = d * 24 * 60 + mt * 60 + rng.randn() * 10
            m = MealEvent(
                time_min  = t_m,
                carbs_g   = carbs + rng.randn() * 5,
                fiber_g   = 3.0,
                protein_g = 20.0,
                fat_g     = 10.0,
                meal_type = mtype,
                gi_value  = gi + rng.randn() * 5,
            )
            all_meals.append(m)

    # Simulate clean glucose trace
    model = GlucoseModel(params, all_meals, use_gi=True)
    total_min = n_days * 24 * 60
    t_grid = np.arange(0, total_min, CGM_INTERVAL_MIN, dtype=float)
    G_clean = model.simulate((0, total_min), t_grid)
    G_noisy = G_clean + rng.randn(len(G_clean)) * noise_std

    day_labels = (t_grid // (24 * 60)).astype(int)

    return {
        "times":      t_grid,
        "glucose":    G_noisy,
        "days":       day_labels,
        "meals":      all_meals,
        "true_params": params,
    }
