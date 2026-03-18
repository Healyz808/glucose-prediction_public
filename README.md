# Minimal Public Release for Scenario 4

This package provides a **minimal public-release implementation** of the core personalised multi-horizon prediction experiment from the manuscript:

**“Integrating glycemic index and meal dynamics into glucose prediction models for type 2 diabetes management”**.

## Scope

This release is intentionally limited to the core **Scenario 4** workflow:

- single-subject glucose prediction
- leave-one-day-out cross-validation (LODOCV)
- 15 / 30 / 45 / 60 minute prediction horizons
- Bayesian optimisation of six model parameters
- RMSE-based summary outputs

This package is **not** a full reproduction of every figure, table, or internal analysis used during manuscript development.

## Files

- `scenario4_min_public.py` — main minimal reproducible demo script
- `run_all_subjects.py` — optional batch runner for multiple subject CSV files
- `requirements.txt` — Python dependencies
- `LICENSE` — license for public release

## Expected Input Format

Each subject CSV file must contain the following columns:

- `Timestamp`
- `Libre GL`
- `Meal Type`
- `GI`
- `Carbs`

Example row:

```text
Timestamp,Libre GL,Meal Type,GI,Carbs
2025-01-01 08:00:00,118.4,Breakfast,55,45
```

## Public Dataset

The study uses the **CGMacros Dataset v1.0.0** from PhysioNet, which is publicly available.

Please note that this public code expects **subject-level CSV files already formatted** with the columns above. Raw PhysioNet files may require preprocessing before use.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Demo

Run a small synthetic example:

```bash
python scenario4_min_public.py --demo --output_dir demo_outputs
```

## Run on One Subject

```bash
python scenario4_min_public.py \
  --input_csv path/to/subject_049.csv \
  --output_dir results_subject_049
```

## Batch Run Across Multiple Subjects

```bash
python run_all_subjects.py \
  --input_dir path/to/formatted_subject_csvs \
  --output_dir batch_results
```

## Outputs

For a single subject, the script generates:

- `scenario4_detailed_results.csv`
- `scenario4_summary_results.csv`

For the batch runner, the script generates:

- `scenario4_all_subjects_detailed.csv`
- `scenario4_all_subjects_summary.csv`

## Notes

- This release focuses on **clarity, portability, and minimal reproducibility**.
- It omits internal exploratory utilities, figure-specific scripts, and manuscript drafting code.
- Numerical results may vary from the final manuscript tables if different preprocessing, subject formatting, or software versions are used.
