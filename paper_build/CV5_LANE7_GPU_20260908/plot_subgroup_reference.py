"""Full retained-survey subgroup interaction reference on common evaluation contexts.

Keeps the original subgroup specification (20 basic controls plus 9 age/sex/
framing terms), distinct from the common23 overall economic model. This pooled
descriptive reference is not an out-of-fold prediction benchmark.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

Q = Path(__file__).resolve().parents[1]
OUT = Q / 'figures'
ROOT = Q.parents[1]
sys.path.insert(0, str(ROOT.parent / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(OUT / 'source_snapshots'))
import section4_results_case_support as C
S = C.S


def main():
    OUT.mkdir(exist_ok=True)
    source_paths = [S.PATH_TRAIN, S.PATH_TEST, S.PATH_PANEL, S.PATH_GRID, S.PATH_HUMAN_SPEC]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    spec = json.loads(S.PATH_HUMAN_SPEC.read_text(encoding='utf-8'))['behavioral_reference_specification']
    # The original retained train + test partitions define the same full CV5 pool.
    profiles = pd.concat([pd.read_parquet(S.PATH_TRAIN, columns=['case_id', 'respondent_id', 'y']),
                          pd.read_parquet(S.PATH_TEST, columns=['case_id', 'respondent_id', 'y'])], ignore_index=True)
    profiles['case_id'] = profiles.case_id.astype(str)
    assert len(profiles) == 11584 and profiles.respondent_id.nunique() == 2169 and profiles.case_id.is_unique
    full_index = pd.read_parquet(Q / 'full_sample_index.parquet')
    assert set(profiles.case_id) == set(full_index.case_id.astype(str))
    np.testing.assert_array_equal(profiles.set_index('case_id').y.sort_index(),
                                  full_index.set_index('case_id').y.sort_index())
    test = pd.read_parquet(S.PATH_TEST, columns=['case_id', 'respondent_id'])
    test['case_id'] = test.case_id.astype(str)
    assert len(test) == 3468 and test.respondent_id.nunique() == 651
    panel = pd.read_parquet(S.PATH_PANEL)
    panel['case_id'] = panel.respondent_id.astype(str) + '_' + panel.distance_band.astype(str)
    assert panel.case_id.is_unique
    cases = S._prepare_human_cases(panel.set_index('case_id').loc[profiles.case_id].reset_index(), spec)
    design = S._add_joint_group_interactions(cases, S._build_human_base_design(cases, spec))
    assert design.shape == (11584, 29)
    y = profiles.y.to_numpy(int)
    fit = sm.Logit(y, design).fit(disp=0, maxiter=600, tol=1e-10, cov_type='cluster',
                                 cov_kwds={'groups': cases.respondent_id.to_numpy()})
    assert fit.mle_retvals['converged'] and np.isfinite(fit.cov_params()).all().all()
    scale = np.sqrt((design * design).mean())
    scaled = sm.Logit(y, design / scale)
    score = float(np.max(np.abs(scaled.score(fit.params * scale))) / len(y))
    eig = float(np.linalg.eigvalsh(-scaled.hessian(fit.params * scale) / len(y)).min())
    assert score < 1e-7 and eig > 0
    tc = S._prepare_human_cases(panel.set_index('case_id').loc[test.case_id].reset_index(), spec)
    td = S._add_joint_group_interactions(tc, S._build_human_base_design(tc, spec))
    order = test.case_id.tolist()
    test['distance_band'] = tc.distance_band.to_numpy()
    indicators = pd.DataFrame({'age50': tc.age.ge(50).astype(float),
                               'female': tc.gender_label.eq('Female').astype(float),
                               'risk': tc.BR_label.eq('RISK').astype(float)})
    group_meta = pd.DataFrame({'respondent_id': tc.respondent_id,
                              'age_group': np.where(indicators.age50.eq(1), 'Age 50+', 'Age <50'),
                              'sex_group': np.where(indicators.female.eq(1), 'Female', 'Male'),
                              'framing_group': np.where(indicators.risk.eq(1), 'RISK', 'BASIC')}).drop_duplicates('respondent_id')
    assert len(group_meta) == 651
    group_meta.to_csv(OUT / 'subgroup_respondent_meta.csv', index=False)
    grid = pd.read_parquet(S.PATH_GRID)
    grid['case_id'] = grid.case_id.astype(str)
    base_lp = td.to_numpy() @ fit.params.to_numpy()
    rungs = {}
    for attr, term in [('Fare', 'fare_AV'), ('IVT', 'ivt_AV')]:
        slopes = np.full(len(tc), fit.params[term])
        for name in indicators:
            slopes += indicators[name].to_numpy() * fit.params[f'{term}_x_{name}']
        xs = C._rung_x(grid, attr, order)
        ps = expit(base_lp[:, None] + slopes[:, None] * (xs - xs[:, 3:4]))
        rungs[attr] = (ps, xs)
    Pf, Xf = rungs['Fare']; Pt, Xt = rungs['IVT']
    frames = C._frames_from_operator(C.HUMAN_E, order, test, n_seed=1,
                                    **C._operator_from_matrices(Pf, Xf, Pt, Xt))
    frames['vot']['model'] = C.HUMAN_V
    for key, frame in frames.items():
        frame.to_parquet(OUT / f'subgroup_human_{key}.parquet', index=False)
    coefficients = pd.DataFrame({'term': fit.params.index, 'coefficient': fit.params,
                                  'cluster_se': fit.bse, 'p_value': fit.pvalues})
    coefficients.to_csv(OUT / 'subgroup_human_coefficients.csv', index=False)
    fit.cov_params().to_csv(OUT / 'subgroup_human_covariance.csv')
    names = list(fit.params.index); tests = []
    for name, dimension in [('age50', 'age_group'), ('female', 'sex_group'), ('risk', 'framing_group')]:
        restriction = np.zeros((2, len(names)))
        restriction[0, names.index('fare_AV_x_' + name)] = 1
        restriction[1, names.index('ivt_AV_x_' + name)] = 1
        joint = fit.wald_test(restriction, scalar=True)
        tests.append({'Dimension': dimension, 'Fare interaction p-value': float(fit.pvalues['fare_AV_x_' + name]),
                      'IVT interaction p-value': float(fit.pvalues['ivt_AV_x_' + name]),
                      'Joint Fare+IVT p-value': float(joint.pvalue)})
    pd.DataFrame(tests).to_csv(OUT / 'subgroup_human_interaction_tests.csv', index=False)
    validation = {'status': 'PASS', 'reference_role': 'pooled full retained-survey descriptive interaction reference',
                  'n_fit_cases': 11584, 'n_fit_respondents': 2169, 'n_evaluation_cases': 3468,
                  'n_evaluation_respondents': 651, 'parameters': list(design.columns), 'n_parameters': 29,
                  'max_scaled_mean_score': score, 'min_scaled_information_eigenvalue': eig,
                  'group_counts': {c: group_meta.groupby(c).size().to_dict() for c in ['age_group', 'sex_group', 'framing_group']},
                  'inputs_sha256': source_hashes, 'cross_fitted': False}
    assert source_hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (OUT / 'subgroup_reference_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
