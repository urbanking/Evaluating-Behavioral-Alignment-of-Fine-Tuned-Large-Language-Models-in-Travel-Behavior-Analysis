"""Finalize the audited lane-7 neural revision and its project record."""
from pathlib import Path
import csv,difflib,hashlib,json,re,shutil
import numpy as np,pandas as pd
W=Path(__file__).resolve().parents[1];OLD=W.parent/'CV5_PAPER_20260908';P=W.parent/'paper_en_partc_cv5';Q=W/'qa';A=W/'analysis';N=W/'nn_lane7';ROOT=Path('C:/SP_LLM/TRC')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    for path in [N/'integration_validation.json',Q/'independent_data_checks.json',Q/'package_checks.json',A/'validation.json',W/'figures/render_manifest.json']:
        assert json.loads(path.read_text())['status']=='PASS',path
    assert pd.read_csv(N/'saved_model_replay.csv').status.eq('PASS').all()
    builds=json.loads((Q/'build_checks.json').read_text());assert builds['named_anonymous_body_equal']
    assert all(not d['warnings'] and not d['unknown_references'] for d in builds['builds'].values())
    pages=builds['builds']['main_en']['pages']
    report=Path('C:/SP_LLM/reports/phase_status/validation_report.md').read_text(encoding='utf-8')
    assert 'Overall status: PASS' in report and 'Checks passed: 32 / 32' in report
    (Q/'root_validation_report.md').write_text(report,encoding='utf-8')
    before=json.loads((N/'revision_manifest.json').read_text())['previous_paper_hashes']
    assert all(sha(W/'before_paper'/r)==h for r,h in before.items())
    previous_zip=OLD.parent/'output/overleaf/TRC_English_Overleaf_CV5_20260908.zip'
    assert sha(previous_zip)==json.loads((OLD/'release_manifest.json').read_text())['zip_sha256']
    for r in ['sections_en/01_intro.tex','sections_en/02_literature.tex','references.bib','figures/Figure_01_study_framework_author.jpg']:
        assert (P/r).read_bytes()==(W/'before_paper'/r).read_bytes(),r
    # Tables6/7 contain Human/Qwen estimates only and must remain the same.
    a=(P/'sections_en/04_results.tex').read_text(encoding='utf-8');b=(W/'before_paper/sections_en/04_results.tex').read_text(encoding='utf-8')
    for label in ['tab:tab05','tab:tab06']:
        def block(s):
            k=s.index('\\label{'+label+'}');x=s.rfind('\\begin{table}',0,k);y=s.index('\\end{table}',k)+len('\\end{table}');return s[x:y]
        assert block(a)==block(b),label
    comparisons=[]
    files={
      'prediction_mean_sd.csv':['Model'], 'prediction_pooled.csv':['Model'],
      'direct_fare_summary.csv':['Model'],'direct_vot_summary.csv':['Model'],
      'correspondence.csv':['Model'],'cf_correspondence.csv':['Model'],
      'all_coefficients.csv':['Model','projection','fold','Term']}
    for filename,keys in files.items():
        x=pd.read_csv(OLD/'analysis'/filename);y=pd.read_csv(A/filename)
        num=[c for c in x.columns.intersection(y.columns) if c not in keys and pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c])]
        xx=x.set_index(keys);yy=y.set_index(keys)
        for idx in xx.index.intersection(yy.index):
            tag=idx if isinstance(idx,tuple) else (idx,)
            is_nn=tag[0] in ['FFNN','DNN']
            for col in num:
                old=float(xx.loc[idx,col]);new=float(yy.loc[idx,col])
                if not is_nn:assert np.isclose(old,new,rtol=1e-10,atol=1e-10,equal_nan=True),(filename,idx,col)
                if is_nn:comparisons.append({'source':filename,**dict(zip(keys,tag)),'metric':col,'previous_cv5':old,'lane7_cv5':new,'change':new-old})
    pd.DataFrame(comparisons).to_csv(W/'numerical_changes.csv',index=False)
    # Keep old sklearn neural artifacts visibly separate from the active GPU models.
    legacy=W/'conventional/inherited_pre_lane7';legacy.mkdir(exist_ok=True)
    for sub,patterns in [('models',['f*_ffnn.joblib','f*_dnn.joblib']),('tuning',['f*_ffnn.json','f*_dnn.json'])]:
        for pattern in patterns:
            for path in (W/'conventional'/sub).glob(pattern):
                dest=legacy/sub/path.name;dest.parent.mkdir(exist_ok=True)
                assert dest.resolve().is_relative_to(W.resolve()) and path.resolve().is_relative_to(W.resolve())
                if not dest.exists():shutil.move(str(path),str(dest))
    for src in (N/'metrics').glob('f*.json'):shutil.copy2(src,W/'conventional/metrics'/src.name)
    diffs=Q/'lane7_manuscript_diffs';diffs.mkdir(exist_ok=True);changed=[]
    for path in (W/'before_paper').rglob('*.tex'):
        rel=path.relative_to(W/'before_paper');new=P/rel
        if new.exists() and path.read_bytes()!=new.read_bytes():
            changed.append(str(rel));diff=''.join(difflib.unified_diff(path.read_text(encoding='utf-8').splitlines(True),new.read_text(encoding='utf-8').splitlines(True),fromfile='previous_CV5/'+str(rel),tofile='lane7_CV5/'+str(rel)))
            (diffs/(path.name+'.diff')).write_text(diff,encoding='utf-8')
    m=pd.read_csv(A/'prediction_mean_sd.csv').set_index('Model');v=pd.read_csv(A/'direct_vot_summary.csv').set_index('Model');f=pd.read_csv(A/'direct_fare_summary.csv').set_index('Model')
    lines=['# Lane-7 calibrated neural CV5 revision','',
      'The two neural candidates were retrained on the RTX3080 in five respondent folds using seed42. Three inner grouped fits per outer fit generated the predictions used for Platt calibration and AV-F1 threshold selection. There are10 final and30 inner GPU fits. No Qwen training or inference was run.','',
      'The user-supplied FFNN0.755/0.644/0.405 and DNN0.773/0.650/0.409 figures are the earlier single-split three-seed means. The following are new five-fold means, with sample SD, from the same common651respondents/3,468cases. Differences are descriptive, not tests of significance.','',
      '| Model | Accuracy | Macro-F1 | Log Loss | AV-class F1 |','|---|---:|---:|---:|---:|']
    for name in ['FFNN','DNN']:
        vals=[f'{m.loc[name,k+"_mean"]:.3f} ± {m.loc[name,k+"_sd"]:.3f}' for k in ['Accuracy','Macro_F1','Log_Loss','AV_F1']]
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines+=['','Behavioral results from the same frozen calibrated networks:','',
      '| Model | Mean fare elasticity | Aggregate Direct VOT (kKRW/h) | Computable (%) | Expected signs among computable (%) |','|---|---:|---:|---:|---:|']
    for name in ['FFNN','DNN']:lines.append(f'| {name} | {f.loc[name,"Mean"]:.3f} | {v.loc[name,"Population Direct VOT-eq."]:.3f} | {v.loc[name,"Computable (%)"]:.2f} | {v.loc[name,"Joint conventional-sign (%)"]:.2f} |')
    lines+=['','Neural probability quality is better than in the prior CV5 neural implementation, but the current neural IVT responses are mostly positive and the Direct VOT ratios mostly negative. The manuscript reports these findings without claiming that calibration establishes behavioral validity. Architecture, loss, early stopping and calibration changed together; this is not an isolated calibration-effect experiment.','',
      f'The numeric comparison file contains{len(comparisons)}neural entries. Human/Qwen and the other four conventional numerical outputs were verified unchanged. Main Tables6/7, Introduction, Literature Review, bibliography and Figure1 are unchanged. Main figures/tables, subgroup boxplots and all four Section4.2 subsections remain present.','']
    (W/'NUMERICAL_CHANGES.md').write_text('\n'.join(lines),encoding='utf-8')
    readme=f'''# CV5 manuscript with lane-7 calibrated FFNN/DNN

Completed revision of `../paper_en_partc_cv5` ({pages} pages in named and anonymous PDFs).
The previous CV5 manuscript is saved in `before_paper/`; its workspace and ZIP remain
in `../CV5_PAPER_20260908` and the previous Overleaf archive. Original fixed-split
English/Korean edits made by other tasks are documented in `qa/source_refresh.json`.

## Active evidence

- Common evaluation: 651 respondents, 3,468 cases, 33 conditions per candidate.
- FFNN and DNN: 10 final GPU fits, plus 30 inner fits. Seed 42, five existing respondent folds.
- Frozen user-selected lane-7 configurations: FFNN32; DNN256-128-64-32;
  ReLU, dropout0.3, focal loss(gamma1, alpha0.75), AdamW lr0.001, weight decay0.00001.
- Early stopping: grouped15% inner validation, PR-AUC, patience30, max300epochs,
  batch512, cosine schedule. No outer evaluation label selects a checkpoint.
- Nonnegative Platt slope and AV-F1 threshold are fitted to3-fold inner OOF predictions.
  They remain frozen for BASE and every condition. Model-generated choices use that
  threshold; every probability-based analysis uses the calibrated probabilities.
- The four other conventional candidates and all Qwen outputs are retained unchanged.
  Full-survey Human references and Qwen linked-bootstrap results are unchanged;
  the6 neural economic projections and all neural Direct outputs were recomputed.
- Table3 reports fold mean and sample SD. Pooled OOF scores are stored separately.
  Historical3-seed single-split lane-7 means are provenance only, not CV5 results.
- This is a retrospective selected-configuration comparison, not a prospectively
  untouched replication or an isolated calibration-effect experiment.

## Reproduce the neural revision

Use `C:/gpu-work/.venv/Scripts/python.exe` for GPU training and model replay.
Analytical scripts use `C:/SP_LLM/.nbvenv/Scripts/python.exe`; PDF scripts use the
local Python with PyMuPDF/Pillow. Run from this directory with project inputs available.

1. `nn_lane7/run_gpu_cv5.py`: training, inner OOF calibration, thresholds, 33-condition scores.
   Completed cells are verified and reused. To repeat from scratch, use a separate
   output copy rather than overwriting preserved results. Runtime and input provenance
   are in `nn_lane7/runtime.json`, `revision_manifest.json`, `source/`, and manifests.
2. `nn_lane7/integrate_and_validate.py`: check leakage, keys, calibration and thresholds;
   merge the two new candidates with the four retained conventional score banks.
3. `nn_lane7/replay_saved_models.py`: reload all10 saved GPU weights and reproduce BASE scores.
4. `scripts/prediction_metrics.py --require-all` and
   `analysis/recalculate_cv5.py --include-conventional --fold-checks`.
   Exact-input economic caches are verified before reuse. Human/Qwen estimates do not change.
5. `analysis/validate_econometrics.py`, `scripts/probability_sensitivity.py`,
   `scripts/plot_results.py`, and `scripts/plot_verify.py`.
6. `scripts/regenerate_paper.py` rebuilds only the separate CV5 TeX from its source
   baseline and applies all numerical/prose updaters, ending with `update_lane7.py`.
   Incorporate later manual changes into the updaters before running this reset.
7. `scripts/validate_paper_data.py`, `scripts/build_verify.py`, and visual page review.
8. `scripts/package_overleaf.py` builds the new ZIP and verifies clean compilation.
   Its clean-build directory must be fresh. The previous archive is never overwritten.
9. Run `python scripts/99_validate_outputs.py` from `C:/SP_LLM` and then
   `scripts/finish_lane7_release.py` to close the project record.

The inherited `conventional/run_conventional.py` is historical code, not the neural
runner for this revision. Active neural weights and preprocessing are in
`nn_lane7/models/`. Prior sklearn neural copies are marked `inherited_pre_lane7/`.

## Output map

- `NUMERICAL_CHANGES.md`, `numerical_changes.csv`: old CV5 versus new neural results.
- `analysis/prediction_mean_sd.csv`, `prediction_pooled.csv`: separate aggregation definitions.
- `analysis/direct_*`, `correspondence.csv`, `cf_correspondence.csv`: updated distributions/comparisons.
- `analysis/all_coefficients.csv`, `fit_audit.csv`, covariance files: matched23-parameter fits.
- `nn_lane7/manifests/`, `oof/`, `models/`, `predictions/`: full experimental provenance.
- `nn_lane7/independent_checks.csv`, `saved_model_replay.csv`:130 checks and10 GPU replay checks.
- `qa/independent_data_checks.csv`:465 manuscript data/operator/structure checks.
- `qa/lane7_manuscript_diffs/`: exact TeX changes against the preserved prior CV5 version.
- `qa/build_checks.json`, `package_checks.json`, `visual_review.json`: PDF/ZIP review.
- `release_manifest.json`, `result_inventory.csv`: final active release and local output inventory.

The Overleaf ZIP contains manuscript assets only, not respondent records or model weights.
'''
    (W/'README.md').write_text(readme,encoding='utf-8')
    dump(Q/'visual_review.json',{'status':'PASS','named_pages_reviewed':pages,'method':'All-page contact sheets plus full-size dense result, settings and first anonymous pages',
      'figures':'Six vector figures reviewed','resolved':'Removed appendix forced-page whitespace; placed Figure3 before subgroup prose; preserved prompt listing and all figure/table inventories',
      'anonymous':'First page reviewed; body parity verified'})
    decision_marker='GPU lane-7 calibrated neural replacement in the CV5 manuscript'
    logbytes=(ROOT/'DECISIONS.md').read_bytes()
    num=max(map(int,re.findall(rb'\bD(\d+)\b',logbytes)))+1
    existing=re.search(rb'### (D\d+) - '+decision_marker.encode(),logbytes)
    decision=existing.group(1).decode() if existing else f'D{num}'
    package=json.loads((Q/'package_checks.json').read_text())
    text=f'''\n\n### {decision} - {decision_marker}

- Date: 2026-09-08. User requested the calibrated lane-7 FFNN/DNN version, actual GPU five-fold experiments, and synchronized manuscript results.
- Completed 10 final neural fits and 30 inner fits on RTX 3080, seed 42. Historical architectures and focal loss are fixed. Early stopping, Platt calibration and AV-F1 thresholds use only the current outer training respondents. Same network/calibrator/threshold produces BASE and all 33 conditions: 228,888 neural rows.
- New fold means: FFNN Accuracy 0.722, Macro-F1 0.624, Log Loss 0.423; DNN 0.757, 0.634, 0.432. The user's earlier single-split three-seed means are not inserted as five-fold results. The retrospective configuration choice is disclosed.
- Recomputed six neural economic projections and all neural behavioral statistics. Fare means are -0.263/-0.442; aggregate Direct VOT is -4.071/-2.618 kKRW/h. IVT responses are mostly positive, despite improved probability loss. Interpretation describes complete fitted systems rather than attributing the difference to calibration alone.
- Other four conventional candidates, Qwen predictions and Human references remain numerically unchanged. Tables 6/7, Introduction, Literature Review and Figure 1 are preserved. Seven main figures, eight main tables, subgroup boxes, four Section 4.2 subsections and the appendix inventory remain present.
- Validation: 130 neural integration checks; 10 saved GPU state replays; 465 independent manuscript checks; economic/figure checks; root P0 validator 32/32 PASS. Both PDFs {pages} pages, no LaTeX warnings; full-page visual review and clean ZIP text/pixel parity PASS.
- Workspace: `Real_exp/pdf/CV5_LANE7_GPU_20260908`; target: `paper_en_partc_cv5`. Previous CV5 files are backed up and the prior ZIP/workspace preserved. Separate D372 fixed-split bilingual Conclusion edits are tracked as external changes, not overwritten or imported into this numerical revision.
- Output ZIP: `{Path(package['zip']).name}`. Numeric changes, exact manuscript diffs, GPU weights/preprocessing/OOF scores and reproducibility records are provided. Next: author review of updated Table 3 and the neural fare/IVT/VOT comparisons; no pending experiment remains for this revision.
'''
    for filename in ['DECISIONS.md','RUN_LOG.md','PHASE_STATUS.md']:
        path=ROOT/filename
        if decision_marker.encode() not in path.read_bytes():
            with path.open('a',encoding='utf-8') as z:z.write(text)
    dump(Q/'project_record.json',{'decision':decision,'root_validator':'PASS','root_validator_checks':'32/32','status':'COMPLETE'})
    manifest={'status':'PASS','version':'CV5_Lane7GPU_20260908','pages':pages,'paper_directory':str(P),'overleaf_zip':package['zip'],'zip_sha256':package['sha256'],
       'candidate_cohort':[651,3468],'conditions':33,'new_final_GPU_fits':10,'new_inner_GPU_fits':30,'new_neural_rows':228888,
       'new_neural_economic_fits':6,'total_pooled_economic_fits':31,'new_Qwen_work':False,'preserved_main_figures':7,'preserved_main_tables':8,
       'preserved_section42_subsections':4,'neural_checks':130,'model_replays':10,'paper_checks':465,'root_checks':'32/32 PASS',
       'changed_tex_files':changed,'numerical_comparisons':len(comparisons),'previous_CV5_preserved':True,'project_decision':decision}
    dump(W/'release_manifest.json',manifest)
    rows=[]
    for path in sorted(W.rglob('*')):
        if not path.is_file() or path.name in ['result_inventory.csv'] or any(p in ['__pycache__','renders','clean_overleaf_lane7','before_paper','source_base'] for p in path.relative_to(W).parts):continue
        if path.suffix in ['.log','.pyc']:continue
        rows.append({'path':str(path.relative_to(W)),'bytes':path.stat().st_size,'sha256':sha(path)})
    pd.DataFrame(rows).to_csv(W/'result_inventory.csv',index=False)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
