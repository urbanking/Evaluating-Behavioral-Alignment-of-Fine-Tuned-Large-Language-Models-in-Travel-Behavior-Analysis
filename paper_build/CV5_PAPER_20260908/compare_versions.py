"""Record numeric version changes, distinguishing the changed estimands."""
from pathlib import Path
import pandas as pd
W=Path(__file__).resolve().parents[1];A=W/'analysis';OLD=W.parent/'INCOME_LINEAR_NO_PURPOSE_20260907'
rows=[]
def compare(old,new,keys,columns,scope,old_context,new_context):
    old=old.set_index(keys);new=new.set_index(keys)
    for idx in old.index.intersection(new.index):
        tag=idx if isinstance(idx,tuple) else (idx,)
        for ko,kn in columns:
            a=float(old.loc[idx,ko]);b=float(new.loc[idx,kn])
            rows.append(dict(scope=scope,**dict(zip(keys,tag)),metric=kn,previous=a,cv5=b,change=b-a,previous_definition=old_context,cv5_definition=new_context))
def main():
    pm=pd.read_csv(W.parent/'REVISION_IMPLEMENTATION_20260908/prediction_audit/metrics_mean.csv')
    pn=pd.read_csv(A/'prediction_mean_sd.csv')
    compare(pm,pn,['Model'],[('Accuracy','Accuracy_mean'),('Macro-F1','Macro_F1_mean'),('Log Loss','Log_Loss_mean'),('AV-F1','AV_F1_mean'),('AV recall','AV_recall_mean')],
            'Table3 prediction','Fixed split; conventional mean across 3 seeds, Qwen one run','Arithmetic mean across 5 common-cohort fold portions; seed42. Not the same aggregation as previous.')
    for filename,columns,scope in [
        ('direct_fare_summary.csv',['Mean','SD','Median','Q1','Q3','Negative (%)','Absolute gap from Human mean'],'Table4 fare distribution'),
        ('direct_vot_summary.csv',['Computable (%)','Fare-negative (%)','IVT-negative (%)','Joint conventional-sign (%)','Population Direct VOT-eq.','Median','Q1','Q3'],'Table5 VOT distribution'),
        ('correspondence.csv',['GC-MNL Fare elasticity','Direct Fare elasticity','Absolute elasticity gap','GC-MNL VOT','Direct VOT-eq.','Absolute VOT gap'],'Table8 BASE correspondence'),
        ('cf_correspondence.csv',['CF-MNL Fare elasticity','Absolute CF-Direct elasticity gap','CF-MNL VOT','Absolute CF-Direct VOT gap'],'Table8 CF correspondence')]:
        old=pd.read_csv(OLD/filename);new=pd.read_csv(A/filename)
        compare(old,new,['Model'],[(c,c) for c in columns],scope,'Fixed-split candidates; development-only human reference','OOF common-cohort candidates; full-survey descriptive human reference')
    old=pd.read_csv(OLD/'fit_audit.csv');new=pd.read_csv(A/'fit_audit.csv').query('fold == 0')
    cols=['n_rows','n_respondents','vot','elasticity','beta_fare_av','se_fare_av','p_fare_av','beta_ivt_av','se_ivt_av','p_ivt_av','beta_wait_av','se_wait_av','p_wait_av']
    compare(old,new,['Model','projection'],[(c,c) for c in cols],'Economic fits','Previous 23-parameter fit','Same 23 parameters; new choices and full human survey')
    co=pd.read_csv(OLD/'all_coefficients.csv');cn=pd.read_csv(A/'all_coefficients.csv').query('fold == 0')
    compare(co,cn,['Model','projection','Term'],[(c,c) for c in ['Coefficient','Cluster SE','p-value']],'All economic coefficients','Previous fit','CV5 fit (not a test of change between fits)')
    data=pd.DataFrame(rows);data.to_csv(W/'numerical_changes.csv',index=False)
    m=pn.set_index('Model');fa=pd.read_csv(A/'direct_fare_summary.csv').set_index('Model');v=pd.read_csv(A/'direct_vot_summary.csv').set_index('Model')
    lines=['# CV5 numerical changes','',
           'The original English and Korean manuscripts are preserved. This is a separate retrospective five-fold version with the same main figures, tables, subgroup plots and appendix structure.','',
           'The CSV records old and new values with their definitions. Differences are descriptive version changes, not significance tests: the training partitions, full-survey human reference, conventional fits, and Table 3 aggregation changed together.','',
           '| Model | Macro-F1 (fold mean ± SD) | Direct fare mean | Aggregate Direct VOT (kKRW/h) | VOT expected signs (%) |','|---|---:|---:|---:|---:|']
    for q in ['Qwen 2B ZS','Qwen 2B SFT','Qwen 9B ZS','Qwen 9B SFT']:
        lines.append(f'| {q} | {m.loc[q,"Macro_F1_mean"]:.3f} ± {m.loc[q,"Macro_F1_sd"]:.3f} | {fa.loc[q,"Mean"]:.3f} | {v.loc[q,"Population Direct VOT-eq."]:.3f} | {v.loc[q,"Joint conventional-sign (%)"]:.2f} |')
    lines+=['','Main changes requiring different wording:',
            '- Both Qwen sizes improve Macro-F1 and AV identification; the 9B Log Loss reduction is small. Their SFT Macro-F1 values are similar.',
            '- Both mean fare elasticities move closer to the full-survey reference, but 2B dispersion increases.',
            '- Both VOT interquartile ranges narrow; expected-sign shares decline at both sizes.',
            '- BASE-to-Direct VOT correspondence improves strongly for 9B, while the 2B scalar gap increases slightly.',
            '- H-MNL IVT p = 0.098 and VOT Fieller set [−1.198, 15.051] kKRW/h; the human time-value reference is uncertain.',
            '- Linked respondent bootstrap supports different endpoints at different scales. It conditions on existing fitted models.',
            '',f'Complete numeric comparison: {len(data):,} rows in `numerical_changes.csv`. Pooled OOF prediction metrics are separately stored in `analysis/prediction_pooled.csv`; they are not substituted for fold means in Table 3.','']
    (W/'NUMERICAL_CHANGES.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Wrote {len(data)} numeric comparisons')
if __name__=='__main__':main()
