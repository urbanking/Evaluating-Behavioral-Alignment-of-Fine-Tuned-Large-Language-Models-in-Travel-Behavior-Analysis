"""Independent key, metric, numerical-operator and preservation checks."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,log_loss

W=Path(__file__).resolve().parents[1];A=W/'analysis';Q=W/'qa';P=W.parent/'paper_en_partc_cv5'
checks=[]
def check(name,ok,detail=''):
    checks.append({'check':name,'status':'PASS' if bool(ok) else 'FAIL','detail':str(detail)})
    assert ok,(name,detail)
def close(name,a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    check(name,np.allclose(a,b,atol=1e-10,rtol=1e-9,equal_nan=True),f'shape={a.shape}')
def main():
    val=json.loads((A/'validation.json').read_text());check('complete economic candidate set',val['complete_all_models'] and val['fit_count']==31)
    e=pd.read_parquet(W/'evaluation_cases.parquet').set_index('case_id')
    bank=pd.concat([pd.read_parquet(W/'qwen_scores.parquet'),pd.read_parquet(W/'conventional/candidate_scores.parquet')],ignore_index=True)
    grid=pd.read_parquet(W/'scenario_grid.parquet')
    expected=pd.MultiIndex.from_frame(grid[['case_id','scenario_id']]).sort_values()
    check('common cohort',len(e)==3468 and e.respondent_id.nunique()==651 and len(expected)==114444)
    check('ten candidate families',bank.Model.nunique()==10)
    metrics=pd.read_csv(A/'prediction_mean_sd.csv').set_index('Model')
    for model,g in bank.groupby('Model'):
        check(model+' complete unique grid',len(g)==114444 and not g.duplicated(['case_id','scenario_id']).any() and pd.MultiIndex.from_frame(g[['case_id','scenario_id']]).sort_values().equals(expected))
        check(model+' finite probabilities',np.isfinite(g.p).all() and g.p.between(0,1).all())
        check(model+' assigned fold',np.array_equal(g.fold.to_numpy(),e.loc[g.case_id,'fold'].to_numpy()))
        vals=[]
        for fold,base in g[g.scenario_id.eq('BASE')].groupby('fold'):
            y=e.loc[base.case_id,'y'].to_numpy();h=base.hard.to_numpy();p=base.p.to_numpy()
            vals.append([accuracy_score(y,h),f1_score(y,h,average='macro'),log_loss(y,np.clip(p,1e-9,1-1e-9),labels=[0,1])])
        close(model+' fold mean',np.mean(vals,axis=0),metrics.loc[model,['Accuracy_mean','Macro_F1_mean','Log_Loss_mean']])
        close(model+' fold sample SD',np.std(vals,axis=0,ddof=1),metrics.loc[model,['Accuracy_sd','Macro_F1_sd','Log_Loss_sd']])
    manifests=[p for p in (W/'conventional/manifests').glob('f*.json') if not p.name.endswith('_features.json')]
    check('thirty completed conventional manifests',len(manifests)==30)
    banned={'case_id','respondent_id','fold','partition','y','human_label','task_id','QdC','QdD','QdE','AV_choice'}
    for p in manifests:
        m=json.loads(p.read_text(encoding='utf-8'));f=json.loads(Path(m['features_path']).read_text(encoding='utf-8'))
        # Audit the exact selected feature list; other JSON metadata may contain excluded names.
        features=f.get('numeric_features',f.get('numeric',[]))+f.get('categorical_features',f.get('categorical',[]))
        if not features:features=f.get('feature_columns',f.get('features',[]))
        check(p.stem+' no metadata predictors',len(features)==72 and not(set(features)&banned),len(features))
        check(p.stem+' no respondent overlap',m['outer_respondent_overlap']==0 and m['inner_group_overlap']==0)
        check(p.stem+' same model BASE/CF',m['same_fit_base_cf_max_difference']<1e-12 and m['seed']==42,m['same_fit_base_cf_max_difference'])
    intervals=pd.read_parquet(A/'direct_elasticity_intervals.parquet')
    cases=pd.read_parquet(A/'direct_elasticity_cases.parquet')
    vot=pd.read_parquet(A/'direct_vot_cases.parquet')
    human=pd.read_parquet(A/'human_scores.parquet');allbank=pd.concat([bank,human],ignore_index=True)
    for model,g in allbank.groupby('Model'):
        ids=sorted(e.index);out={}
        for prefix,xcol in [('F','fare_cf'),('R','ride_cf')]:
            ladder=[prefix+'_M3',prefix+'_M2',prefix+'_M1','BASE',prefix+'_P1',prefix+'_P2',prefix+'_P3']
            probs=g.pivot(index='case_id',columns='scenario_id',values='p').loc[ids,ladder].to_numpy()
            x=grid.pivot(index='case_id',columns='scenario_id',values=xcol).loc[ids,ladder].to_numpy()
            dp=np.diff(probs,axis=1);dx=np.diff(x,axis=1)
            out[prefix]=np.mean(dp/dx,axis=1)
            if prefix=='F':
                with np.errstate(divide='ignore',invalid='ignore'):
                    arc=(2*dp/(probs[:,1:]+probs[:,:-1]))/(2*dx/(x[:,1:]+x[:,:-1]))
                stored=intervals[intervals.system.eq(model)].pivot(index='case_id',columns='interval',values='elasticity').loc[ids].to_numpy()
                close(model+' 20808 arc intervals',arc,stored)
                c=cases[cases.system.eq(model)].set_index('case_id').loc[ids]
                close(model+' Figure6 case means',arc.mean(axis=1),c.elasticity)
        vm='Human same-CF reference' if model=='Human reference' else model
        v=vot[vot.model.eq(vm)].set_index('case_id').loc[ids]
        close(model+' six-pair DF',out['F'],v.raw_DF);close(model+' six-pair DT',out['R'],v.raw_DT)
        comp=np.isfinite(out['F'])&np.isfinite(out['R'])&(np.abs(out['F'])>=1e-4)
        check(model+' VOT computability',np.array_equal(comp,v.raw_computable.to_numpy()))
        with np.errstate(divide='ignore',invalid='ignore'):ratio=np.where(comp,60*out['R']/out['F'],np.nan)
        close(model+' case VOT',ratio,v.raw_vot)
    fit=pd.read_csv(A/'fit_audit.csv');co=pd.read_csv(A/'all_coefficients.csv')
    for _,f in fit.iterrows():
        b=co[(co.Model==f.Model)&(co.projection==f.projection)&(co.fold==0)].set_index('Term')
        check(f'{f.Model} {f.projection} shared 23 terms',len(b)==23 and f.n_parameters==23)
        close(f'{f.Model} {f.projection} VOT ratio',60*b.loc['ivt_AV','Coefficient']/b.loc['fare_AV','Coefficient'],f.vot)
        for attr in ['fare','ivt','wait']:
            close(f'{f.Model} {f.projection} {attr} SE',b.loc[attr+'_AV','Cluster SE'],f['se_'+attr+'_av'])
            close(f'{f.Model} {f.projection} {attr} p',b.loc[attr+'_AV','p-value'],f['p_'+attr+'_av'])
    mainfiles=[P/'sections_en'/n for n in ['01_intro.tex','02_literature.tex','03_method.tex','04_results.tex','05_06_discussion_conclusion.tex']]
    maintext='\n'.join(p.read_text(encoding='utf-8') for p in mainfiles);app=(P/'sections_en/appendix.tex').read_text(encoding='utf-8')
    check('seven main figures retained',maintext.count('\\begin{figure}')==7)
    check('eight main tables retained',maintext.count('\\begin{table}')==8)
    check('four appendix tables retained',app.count('\\begin{table}')==4)
    sec=(P/'sections_en/04_results.tex').read_text(encoding='utf-8');b=sec[sec.index('\\label{sec:results-direct}'):sec.index('\\label{sec:results-correspondence}')]
    check('four 4.2 subsections retained',b.count('\\subsubsection{')==4)
    check('subgroup figures remain main',all('\\label{'+label+'}' in sec for label in ['fig:fig03','fig:fig05']))
    oldhash=json.loads((Q/'original_hashes.json').read_text());mismatch=[]
    external=json.loads((Q/'source_refresh.json').read_text())['externally_changed_files'] if (Q/'source_refresh.json').exists() else []
    for f,h in oldhash.items():
        if f in external:continue
        p=W.parent/f
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:mismatch.append(f)
    check('original files outside concurrent approved revision preserved',not mismatch,f'{len(oldhash)-len(external)} unchanged; {len(external)} separately documented external edits; {mismatch}')
    check('author Figure1 source preserved',(P/'figures/Figure_01_study_framework_author.jpg').read_bytes()==(W.parent/'paper_en_partc/figures/Figure_01_study_framework_author.jpg').read_bytes())
    pd.DataFrame(checks).to_csv(Q/'independent_data_checks.csv',index=False)
    (Q/'independent_data_checks.json').write_text(json.dumps({'status':'PASS','checks':len(checks),'original_files_unchanged':len(oldhash)-len(external),'separate_external_revision_artifacts':len(external),'candidate_rows':len(bank)},indent=2))
    print(f'PASS: {len(checks)} independent checks')

if __name__=='__main__':
    try:main()
    except Exception:
        pd.DataFrame(checks).to_csv(Q/'independent_data_checks.csv',index=False)
        raise
