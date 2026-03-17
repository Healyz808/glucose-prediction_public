"""
Evaluation Metrics & Multi-Horizon Prediction
==============================================
Implements Scenarios 2–5 from Section 3 and all evaluation metrics
(Section 2g) of the paper.

Metrics:
  - RMSE  (Eq. 2.12 / 2.14)
  - TIR   (Eq. 2.15)
  - TAR   (Eq. 2.16)
  - TBR   (Eq. 2.17)

Experiments:
  - Cross-subject 24h fitting (LOSOCV)
  - Personalised 24h fitting (LODOCV)
  - Personalised multi-horizon prediction
"""

import numpy as np
from typing import List, Dict, Optional
from models.glucose_model import SubjectParameters, GlucoseModel, MealEvent


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE. Eq. 2.14."""
    mask = ~np.isnan(predicted) & ~np.isnan(observed)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((observed[mask] - predicted[mask]) ** 2)))


def mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error (supplementary metric)."""
    mask = ~np.isnan(predicted) & ~np.isnan(observed)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(observed[mask] - predicted[mask])))


def tir(glucose: np.ndarray, lo: float = 70.0, hi: float = 180.0) -> float:
    """Time in range [lo, hi] mg/dL. Eq. 2.15."""
    return float(np.mean((glucose >= lo) & (glucose <= hi)) * 100.0)


def tar(glucose: np.ndarray, hi: float = 180.0) -> float:
    """Time above range. Eq. 2.16."""
    return float(np.mean(glucose > hi) * 100.0)


def tbr(glucose: np.ndarray, lo: float = 70.0) -> float:
    """Time below range. Eq. 2.17."""
    return float(np.mean(glucose < lo) * 100.0)


def glucose_metrics(glucose: np.ndarray) -> Dict[str, float]:
    """Return all clinical metrics for a glucose trace."""
    return {
        "TIR": tir(glucose),
        "TAR": tar(glucose),
        "TBR": tbr(glucose),
        "mean_glucose": float(np.mean(glucose)),
        "std_glucose":  float(np.std(glucose)),
    }


# ─────────────────────────────────────────────
# 24-hour trajectory fitting evaluation
# ─────────────────────────────────────────────

def evaluate_24h_fit(params: SubjectParameters,
                     meals: List[MealEvent],
                     cgm_times: np.ndarray,
                     cgm_glucose: np.ndarray,
                     use_gi: bool = True) -> Dict[str, float]:
    """
    Reconstruct 24h glucose trajectory and compute RMSE.
    Used for Scenarios 2 and 3.
    """
    model = GlucoseModel(params, meals, use_gi=use_gi)
    G0    = cgm_glucose[0]
    G_pred = model.simulate(
        t_span=(cgm_times[0], cgm_times[-1]),
        t_eval=cgm_times,
        G0=G0
    )
    return {
        "rmse":   rmse(cgm_glucose, G_pred),
        "mae":    mae(cgm_glucose, G_pred),
        "G_pred": G_pred,
    }


# ─────────────────────────────────────────────
# Multi-horizon rolling prediction  (Scenario 4)
# ─────────────────────────────────────────────

PREDICTION_HORIZONS = [15, 30, 45, 60]   # minutes


def rolling_prediction(params: SubjectParameters,
                        meals: List[MealEvent],
                        cgm_times: np.ndarray,
                        cgm_glucose: np.ndarray,
                        horizon_min: int = 30,
                        use_gi: bool = True) -> np.ndarray:
    """
    Rolling prediction at a fixed horizon.
    At each CGM step t_k, re-initialise state from observed G(t_k)
    and integrate forward by horizon_min minutes.
    Returns predicted glucose values (same length as cgm_times).
    """
    model = GlucoseModel(params, meals, use_gi=use_gi)
    preds = np.full(len(cgm_times), np.nan)

    for i, t0 in enumerate(cgm_times):
        t1 = t0 + horizon_min
        if t1 > cgm_times[-1]:
            break
        G0     = cgm_glucose[i]
        t_eval = np.array([t0, t1])
        try:
            G_traj = model.simulate((t0, t1), t_eval, G0=G0)
            preds[i] = G_traj[-1]
        except Exception:
            pass

    return preds


def multi_horizon_evaluation(params: SubjectParameters,
                               meals: List[MealEvent],
                               cgm_times: np.ndarray,
                               cgm_glucose: np.ndarray,
                               horizons: List[int] = None,
                               use_gi: bool = True) -> Dict[int, Dict[str, float]]:
    """
    Evaluate rolling predictions across multiple horizons.
    Returns dict: {horizon_min: {"rmse": ..., "mae": ..., "preds": ...}}
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS

    results = {}
    for h in horizons:
        preds = rolling_prediction(params, meals, cgm_times, cgm_glucose,
                                   horizon_min=h, use_gi=use_gi)
        # Align ground truth to predicted indices
        valid = ~np.isnan(preds)
        obs   = cgm_glucose[valid]
        pred  = preds[valid]
        results[h] = {
            "rmse":  rmse(obs, pred),
            "mae":   mae(obs, pred),
            "preds": preds,
        }
    return results


# ─────────────────────────────────────────────
# Cohort-level summary
# ─────────────────────────────────────────────

def cohort_summary(subject_results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Aggregate per-subject RMSE into cohort mean ± std.
    subject_results: {subject_id: {"mean_rmse": ..., "std_rmse": ...}}
    """
    rmse_values = [v["mean_rmse"] for v in subject_results.values()
                   if not np.isnan(v.get("mean_rmse", np.nan))]
    if not rmse_values:
        return {"mean_rmse": np.nan, "std_rmse": np.nan}
    return {
        "mean_rmse": float(np.mean(rmse_values)),
        "std_rmse":  float(np.std(rmse_values)),
        "n_subjects": len(rmse_values),
    }


def horizon_cohort_summary(all_subject_horizon_results: List[Dict[int, Dict]],
                            horizons: List[int] = None) -> Dict[int, Dict]:
    """
    Aggregate multi-horizon RMSE across subjects.
    Returns {horizon: {"mean_rmse": ..., "std_rmse": ...}}.
    """
    if horizons is None:
        horizons = PREDICTION_HORIZONS
    summary = {}
    for h in horizons:
        rmse_list = [r[h]["rmse"] for r in all_subject_horizon_results
                     if h in r and not np.isnan(r[h]["rmse"])]
        summary[h] = {
            "mean_rmse": float(np.mean(rmse_list)) if rmse_list else np.nan,
            "std_rmse":  float(np.std(rmse_list))  if rmse_list else np.nan,
        }
    return summary
