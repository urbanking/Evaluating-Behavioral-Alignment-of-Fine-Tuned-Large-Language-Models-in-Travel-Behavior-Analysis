"""Read-only source intake for the isolated CV5 manuscript; no inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.special import expit

TRC = Path('C:/SP_LLM/TRC')
WORK = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    folds = pd.read_csv(TRC/'data/splits/cv5_folds.csv', dtype={'respondent_id': str})
    pool = pd.concat([pd.read_parquet(TRC/f'data/model_inputs/HUMAN_retro_{s}.parquet') for s in ['development','test']], ignore_index=True)
    pool['respondent_id'] = pool.respondent_id.astype(str)
    pool['case_id'] = pool.case_id.astype(str)
    pool = pool.merge(folds, on='respondent_id', validate='many_to_one')
    test = pd.read_parquet(TRC/'data/model_inputs/HUMAN_retro_test.parquet', columns=['case_id','respondent_id','y'])
    test['case_id'] = test.case_id.astype(str)
    test['respondent_id'] = test.respondent_id.astype(str)
    test = test.merge(folds, on='respondent_id', validate='many_to_one')
    assert len(pool)==11584 and pool.respondent_id.nunique()==2169
    assert len(test)==3468 and test.respondent_id.nunique()==651
    assert folds.respondent_id.is_unique and set(folds.fold)==set(range(1,6))
    test.to_parquet(WORK/'evaluation_cases.parquet', index=False)
    pool[['case_id','respondent_id','y','fold']].to_parquet(WORK/'full_sample_index.parquet', index=False)
    fmap=test.set_index('case_id').fold
    grid=pd.read_parquet(TRC/'data/scenarios/cv5_scenario_grid.parquet')
    grid['case_id']=grid.case_id.astype(str)
    grid=grid[grid.case_id.isin(set(test.case_id))].copy()
    expected=pd.MultiIndex.from_frame(grid[['case_id','scenario_id']])
    assert len(expected)==114444 and expected.is_unique
    grid.to_parquet(WORK/'scenario_grid.parquet',index=False)
    specs={
      'Qwen 2B ZS':[TRC/f'predictions/cv5_llm/qwen_mid_zs_f{f}_scenarios.parquet' for f in range(1,6)],
      'Qwen 2B SFT':[TRC/f'predictions/cv5_llm/qwen_mid_f{f}_scenarios.parquet' for f in range(1,6)],
      'Qwen 9B ZS':[TRC/f'predictions/cv5_llm_zs_grid/qwen_p{p}.parquet' for p in [1,2]],
      'Qwen 9B SFT':[TRC/f'predictions/cv5_llm_grid/qwen_f{f}.parquet' for f in range(1,6)]}
    parts=[]; inventory=[]; sources=[]
    for model,files in specs.items():
      d=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
      d['case_id']=d.case_id.astype(str)
      d=d[d.case_id.isin(set(test.case_id))].copy()
      actual=pd.MultiIndex.from_frame(d[['case_id','scenario_id']])
      assert len(d)==114444 and actual.is_unique and actual.difference(expected).empty and expected.difference(actual).empty
      if 'fold' in d: assert d.fold.astype(int).eq(d.case_id.map(fmap)).all()
      d['fold']=d.case_id.map(fmap).astype(int)
      req=['q_av','q_keep','margin','candidate_mass']
      assert np.isfinite(d[req].to_numpy(float)).all() and d.free_argmax.notna().all()
      assert np.allclose(d.margin,d.q_av-d.q_keep,atol=1e-7,rtol=0)
      m=d.margin.to_numpy(float)
      d['p']=expit(m)
      d['hard']=np.where(m>0,1,np.where(m<0,0,d.free_argmax.eq('AV').to_numpy(int)))
      d['Model']=model
      parts.append(d[['case_id','scenario_id','Model','p','hard','fold','margin','free_argmax','candidate_mass']])
      inventory.append(dict(Model=model,rows=len(d),cases=d.case_id.nunique(),scenarios=d.scenario_id.nunique(),folds=5,duplicate_keys=0,missing_required=0,nonfinite=0,candidate_mass_min=float(d.candidate_mass.min()),ties=int((m==0).sum())))
      sources.extend(dict(Model=model,path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files)
    pd.concat(parts,ignore_index=True).to_parquet(WORK/'qwen_scores.parquet',index=False)
    pd.DataFrame(inventory).to_csv(WORK/'qa/qwen_inventory.csv',index=False)
    (WORK/'qa/qwen_sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    rows=[]
    for f in range(1,6):
      tr=pool[pool.fold.ne(f)];te=pool[pool.fold.eq(f)];ev=test[test.fold.eq(f)]
      assert set(tr.respondent_id).isdisjoint(set(te.respondent_id))
      rows.append(dict(fold=f,train_cases=len(tr),train_respondents=tr.respondent_id.nunique(),train_av=int(tr.y.sum()),train_keep=int((1-tr.y).sum()),heldout_cases=len(te),heldout_respondents=te.respondent_id.nunique(),evaluation_cases=len(ev),evaluation_respondents=ev.respondent_id.nunique(),evaluation_av=int(ev.y.sum()),train_prior_logodds=float(np.log(tr.y.mean()/(1-tr.y.mean())))))
    pd.DataFrame(rows).to_csv(WORK/'qa/fold_counts.csv',index=False)
    print(pd.DataFrame(inventory).to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
