# Qwen 9B zero-shot 33-scenario grid receipt (2026-09-07)

This folder preserves the scoring code and logs for the completed Qwen 9B
zero-shot counterfactual grid retrieved from the GV100 host. The two merged
prediction parts are stored under
`C:/SP_LLM/TRC/predictions/cv5_llm_zs_grid/`.

## Coverage and data-quality gate

- Rows: 114,444 / 114,444.
- Cases: 3,468 / 3,468 official-test cases.
- Scenarios: 33 / 33; each scenario has exactly 3,468 rows.
- Part overlap: 0 cases; duplicate `(case_id, scenario_id)` keys: 0.
- Missing/non-finite required scores: 0.
- Candidate mass: minimum 0.9887384, mean 0.9956717, maximum 0.9996405;
  rows below 0.5: 0.
- `free_argmax`: Keep 101,872; AV 12,572.
- `margin == q_av - q_keep` exactly; `p_av_raw + p_keep_raw` agrees with
  `candidate_mass` within 1.28e-7.
- The case/scenario key set exactly matches
  `data/scenarios/cv5_scenario_grid.parquet` restricted to the 3,468
  official-test cases.

## Remote/local SHA-256 verification

| File | Bytes | SHA-256 |
|---|---:|---|
| `predictions/cv5_llm_zs_grid/qwen_p1.parquet` | 2,644,789 | `02AC31358442838198567584DDD93DAC4EA74D55CCEC5527228120F116F088D0` |
| `predictions/cv5_llm_zs_grid/qwen_p2.parquet` | 2,644,163 | `AADD69E4EC5758210B65D09A0A4205E871352CEADA2BE626BA365AB5A74108A2` |
| `cv5_score_zeroshot_grid.py` | 5,816 | `D51113AB9581341877C833187BB71DB4A4E34E15224FF9A36B2DA0F682005279` |
| `run_cv5_zs_grid.ps1` | 1,290 | `E49155D6A16EF439358E20F7DB1EA09A0790BD313CC2C40CB6072BFEB8BD8874` |
| `cv5_zs_grid_chain.ps1` | 1,385 | `B8E0E5064CB21B7C60E835BD226D2F3D93A68EE18B6E5A9B7701FFCB7D8AB92A` |
| `cv5zsgrid_qwen_p1.out` | 100 | `A5640AF84BC1F912F29CEDE4511FE67EF2950432E7E879AE70ECA34ECB9BF658` |
| `cv5zsgrid_qwen_p1_r.out` | 8,956 | `9CCF3B285C5D4359776AC63D7FA10D30B743C7DD030373C5B1B526CE933BDCC2` |
| `cv5zsgrid_qwen_p2.out` | 100 | `5A5F8ACE2AFFC4E75521F862E961729BF2C61B86A5536B7A787D30A9EE028D55` |
| `cv5zsgrid_qwen_p2_r.out` | 8,956 | `6C94D8D79A2A9F02B5D652848CE100F265EC7DE0A5DC8B50E5B568CA838C19DF` |
| `cv5zsgrid_chain.out` | 274 | `CFFA5E3FFD0737EE1CAE0BF75512A24665A834F607DBFBE0B4CE63C88BE8F81F` |

Every remote/local byte count and SHA-256 value matches.

## Repeated BASE inference

The full-grid BASE and the separately completed 11,584-case BASE scoring run
share all 3,468 official-test cases. They differ on one hard label, with mean
absolute pair-normalized probability difference 0.000825 and maximum 0.019505.
The all-model full-grid package uses the BASE rows embedded in this 33-scenario
run for both prediction and behavioural calculations. This avoids combining a
BASE from one inference pass with counterfactuals from another pass.
