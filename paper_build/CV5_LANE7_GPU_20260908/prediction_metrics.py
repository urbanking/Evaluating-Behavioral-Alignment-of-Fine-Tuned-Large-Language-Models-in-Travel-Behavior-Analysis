"""Fold mean/SD and pooled metrics from the same BASE rows used for CF analysis."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

WORK=Path(__file__).resolve().parents[1]
ORDER=['Panel Logit','SVM','Random Forest','XGBoost','FFNN','DNN','Qwen 2B ZS','Qwen 2B SFT','Qwen 9B ZS','Qwen 9B SFT']
def calculate(d):
    y=d.y.to_numpy(int);h=d.hard.to_numpy(int);p=d.p.to_numpy(float)
    return dict(Accuracy=accuracy_score(y,h),Macro_F1=f1_score(y,h,average='macro',zero_division=0),Log_Loss=log_loss(y,np.clip(p,1e-9,1-1e-9),labels=[0,1]),AV_F1=f1_score(y,h,zero_division=0),AV_recall=recall_score(y,h,zero_division=0),AV_precision=precision_score(y,h,zero_division=0),predicted_AV_share=float(h.mean()),observed_AV_share=float(y.mean()),n_cases=len(d),n_respondents=d.respondent_id.nunique())

def main():
    frames=[pd.read_parquet(WORK/'qwen_scores.parquet')]
    c=WORK/'conventional/candidate_scores.parquet'
    if c.exists(): frames.append(pd.read_parquet(c))
    elif '--require-all' in sys.argv: raise FileNotFoundError(c)
    bank=pd.concat(frames,ignore_index=True)
    test=pd.read_parquet(WORK/'evaluation_cases.parquet')
    base=bank.loc[bank.scenario_id.eq('BASE'),['Model','case_id','p','hard','fold']].merge(test[['case_id','respondent_id','y']],on='case_id',validate='many_to_one')
    assert not base.duplicated(['Model','case_id']).any()
    rows=[];pooled=[]
    for model,d in base.groupby('Model'):
        assert len(d)==3468 and set(d.fold)==set(range(1,6))
        pooled.append(dict(Model=model,**calculate(d)))
        for f,g in d.groupby('fold'):rows.append(dict(Model=model,fold=int(f),**calculate(g)))
    folds=pd.DataFrame(rows);pool=pd.DataFrame(pooled)
    metrics=['Accuracy','Macro_F1','Log_Loss','AV_F1','AV_recall','AV_precision','predicted_AV_share']
    summary=[]
    for model in ORDER:
        d=folds[folds.Model.eq(model)]
        if d.empty:continue
        row={'Model':model}
        for m in metrics:row[m+'_mean']=d[m].mean();row[m+'_sd']=d[m].std(ddof=1)
        summary.append(row)
    pd.DataFrame(summary).to_csv(WORK/'analysis/prediction_mean_sd.csv',index=False)
    folds.to_csv(WORK/'analysis/prediction_by_fold.csv',index=False)
    pool.to_csv(WORK/'analysis/prediction_pooled.csv',index=False)
    train=pd.read_csv(WORK/'qa/fold_counts.csv').set_index('fold')
    b=test.copy();b['p']=b.fold.map(train.train_av/train.train_cases);b['hard']=0
    baseline={'pooled':calculate(b),'folds':[dict(fold=int(f),**calculate(g)) for f,g in b.groupby('fold')]}
    (WORK/'analysis/prediction_baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
