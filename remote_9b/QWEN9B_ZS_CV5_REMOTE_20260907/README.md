# Qwen 9B zero-shot CV5 receipt (2026-09-07)

This folder records the completed Qwen 9B zero-shot BASE scoring run retrieved
from the GV100 host. The prediction shards themselves are stored under
`C:/SP_LLM/TRC/predictions/cv5_llm_zs/`; this receipt keeps the exact scoring
script and terminal logs that produced them.

## Run scope and validation

- Model family/state: Qwen 9B, zero-shot, HUMAN world, BASE scenario.
- Cases: 11,584 unique cases, split into two disjoint 5,792-row shards.
- Fold labels: all folds 1--5 from `data/splits/cv5_folds.csv` are present.
- Duplicate `(case_id, scenario_id)` keys: 0.
- Missing or non-finite required scores: 0.
- Candidate probability mass: minimum 0.9888700, mean 0.9958017,
  maximum 0.9996957.
- Score identity check: `margin == q_av - q_keep` exactly.

## Retrieved files and SHA-256

| File | Bytes | SHA-256 |
|---|---:|---|
| `predictions/cv5_llm_zs/qwen_base_p1.parquet` | 199,648 | `582A956F6B90F3603CA86A82B923AEE513ADA683BA718B0F6DB4D66B8D542287` |
| `predictions/cv5_llm_zs/qwen_base_p2.parquet` | 199,637 | `9B26ECC273C707BB7E54F0D67EBF78F656D7711E7CFACFDD71224BE282E9405A` |
| `cv5zs_qwen_p1_r.out` | 855 | `27D6002A5644E9D04321066FCDF6927D9058653777D834485CCC89D00BE992CE` |
| `cv5zs_qwen_p2_r.out` | 855 | `AD2FFB65CAAAE21B18B2660A91453DA6CAC8B6135852C5335FBE5595FFBED01A` |
| `cv5_score_zeroshot.py` | 6,715 | `A216043D56923F216E41E6EEB478FC20E5AADF46F89F8D1320F3EA63F704C50C` |

The remote and local SHA-256 values were checked and matched for every file.

## Paper-facing comparison

On the original 3,468 official-test cases, the retrieved BASE run gives:

- AV-class F1: 0.2803115.
- Accuracy: 0.8134371.
- ROC-AUC: 0.6538927.
- PR-AUC: 0.3322646.

Against the frozen official-test BASE output, there is one hard-label mismatch,
mean absolute probability difference 0.0009306, and maximum absolute probability
difference 0.0220283. The common-fold package therefore uses the newly retrieved
BASE values for prediction metrics. It intentionally retains the frozen,
internally consistent 33-scenario grid for behavioural response calculations;
new BASE scores are not mixed with older counterfactual scenario scores.

Compiled outputs and the machine-readable comparison are in
`SECTION4_RESULTS_CV5_ALL_MODELS_20260907/QWEN9B_ZS_BASE_REPRODUCIBILITY.json`.
