# Evaluating Behavioral Alignment of Fine-Tuned Large Language Models in Travel Behavior Analysis

Source code for the paper *"Beyond Choice Prediction: Evaluating Behavioral Alignment of Fine-Tuned Large Language Models in Travel Behavior Analysis"* (submitted to Transportation Research Part C).

This repository contains **code only**. The stated-preference survey data, model weights and adapters, per-case predictions, counterfactual score banks, result tables, figures, and run logs are not included. The code is published as it was executed; machine-specific paths are documented below rather than rewritten.

## What is here

| Folder | Content |
| --- | --- |
| `configs/` | `trc_experiment.yaml` (experiment contract: worlds, seeds, ladders, LLM settings) and `trc_feature_schemas.yaml` (feature schema, exclusions, post-choice attitude list). All constants are read from these two files. |
| `pipeline/` | Core modules. Data preparation (`p01`–`p04b`), the five-fold scenario grid and truth (`p09b`, `p10b`), prompt rendering (`p14_llm_prompts.py`, `p14b_codebook_labels.py`), QLoRA supervised fine-tuning (`p16_sft_runs.py`), Platt calibration (`p17_calibration.py`), bulk scoring (`p18_bulk_inference.py`), the shared LLM runtime (`llm_runtime.py`), tabular candidates (`tabular.py`), metrics and analysis helpers, and the modules they import. |
| `scripts/` | Five-fold (CV5) lane. Tabular candidates (`cv5_tabular*.py`), Qwen 2B SFT fold scoring at BASE and over the 33-condition grid (`cv5_score_fold*.py`, chain launchers `*.ps1`), Qwen 2B zero-shot scoring (`cv5_score_zs*.py`), summaries and score-bank assembly (`cv5_llm_summary.py`, `cv5_scenarios_summary.py`, `build_cv5_*`). |
| `remote_9b/` | Qwen 9B five-fold scoring scripts that ran on a separate GPU host: SFT fold scoring, zero-shot BASE, and the zero-shot 33-condition grid. Launch wrappers with host-specific settings are omitted. |
| `neural_lanes/` | The FFNN/DNN redesign lanes (PyTorch, RTX 3080): PR-AUC selection, fitted decision threshold, Platt calibration, nested preprocessing. The configuration fixed from these single-split lanes is the one re-implemented inside the five-fold protocol by `paper_build/CV5_LANE7_GPU_20260908/`. |
| `paper_build/` | Scripts that compute the manuscript tables and figures from the score banks and write the LaTeX assets: `CV5_PAPER_20260908/` and `CV5_LANE7_GPU_20260908/` (the latter is the version behind the current manuscript), plus the shared support modules in `tools/`. |

## Environment

Two Python environments were used and are reflected in `requirements.txt`.

- **GPU environment** (fine-tuning, scoring, neural lanes): Python 3.13, `torch 2.11.0+cu128`, `transformers 5.8.0`, `peft 0.19.1`, `bitsandbytes`, `scikit-learn 1.9.0`, `pandas 3.0.2`, `numpy 2.4.3`. Located at `C:\gpu-work\.venv` on the original machine and referenced by that path in the two `scripts/*.ps1` launchers.
- **Paper-build environment** (tables, figures, PDF checks): Python 3.13 with `PyMuPDF`, `pypdf`, `statsmodels`, `matplotlib`, `scipy`.

`flash-linear-attention` is optional; `llm_runtime.py` falls back to the PyTorch implementation when it is absent, which is how the reported runs were made on the RTX 3080.

## Paths and settings you must adapt

- **Project root.** 27 files set `ROOT = Path(r"C:\SP_LLM\TRC")` or resolve it from their own location. Place this repository at that path, or edit `ROOT` in `pipeline/common.py`, the `neural_lanes/*/*.py` files, and `paper_build/*/*.py`.
- **Model weights.** Base checkpoints are resolved through the `TRC_MODEL_ROOT` environment variable (see `pipeline/common.py` and `pipeline/llm_runtime.py`). The runs used `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-9B` snapshots under that root.
- **GPU selection.** `CUDA_VISIBLE_DEVICES` as in the `.ps1` launchers. `FOR_DISABLE_CONSOLE_CTRL_HANDLER=1` is set in the launchers because the Intel Fortran runtime under numpy/MKL aborted one fold on a console control event.

## Reproduction order

1. `pipeline/p01_prepare_panel.py` → `p01b` → `p02` → `p03` → `p04` → `p04b` (panel, spatial context, domain, information conditions, support map, band ladders).
2. `pipeline/p09b_cv5_grid.py`, `pipeline/p10b_cv5_truth.py` (33-condition grid over all 11,584 cases; five respondent-level folds).
3. `pipeline/p14_llm_prompts.py`, `pipeline/p14b_codebook_labels.py` (prompt rendering and codebook labels).
4. `scripts/cv5_tabular.py`, `scripts/cv5_tabular_scenarios.py` (Pooled logit, Random Forest, XGBoost per fold).
5. `pipeline/p16_sft_runs.py` per fold (QLoRA SFT; configuration in Appendix A.2 of the paper).
6. `scripts/cv5_score_fold.py`, `scripts/cv5_score_fold_scenarios.py` (2B SFT), `scripts/cv5_score_zs.py`, `scripts/cv5_score_zs_scenarios.py` (2B zero-shot), `remote_9b/` (9B).
7. `neural_lanes/` and `paper_build/CV5_LANE7_GPU_20260908/` (FFNN/DNN inside the five-fold protocol; Platt calibration and AV-F1 threshold fitted on inner out-of-fold predictions).
8. `paper_build/CV5_LANE7_GPU_20260908/` scripts in the order `prepare_qwen_bank.py` → `prediction_metrics.py` → `update_tables.py` / `plot_results.py` → `regenerate_paper.py` → `build_verify.py`.

## Notes on the zero-shot prompts

The zero-shot prompt styles are frozen per model size (`zeroshot` for 2B, `ab` with a closed thinking block for 9B). The registered 2B zero-shot instruction ends with a fixed answer-format paragraph; `scripts/cv5_score_zs.py` reconstructs it verbatim and verifies the reconstruction byte-for-byte against the frozen development prompts before scoring. Without that paragraph the 2B model does not answer at the scored position (candidate mass 0.03 instead of 0.99), so this check is not optional.

## Data availability

The survey microdata (2,178 respondents, Seoul Capital Area) are not distributed with this code. Fold assignments and fitted calibration coefficients are derived from those data and are therefore also not included here; contact the authors for access under the survey's terms.

## Citation

Citation details will be added on publication.
