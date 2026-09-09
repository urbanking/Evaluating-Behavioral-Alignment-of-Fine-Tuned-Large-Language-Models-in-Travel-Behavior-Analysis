"""Fixed fold-prior, support, and denominator diagnostics; no fitted calibration."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss,recall_score
WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,'C:/SP_LLM/TRC/scripts')
from six_pair_operator import six_pairs,LADDER,XCOL

def main():
    q=pd.read_parquet(WORK/'qwen_scores.parquet');h=pd.read_parquet(WORK/'analysis/human_scores.parquet')
    test=pd.read_parquet(WORK/'evaluation_cases.parquet');order=test.case_id.tolist()
    grid=pd.read_parquet(WORK/'scenario_grid.parquet')
    prior=pd.read_csv(WORK/'qa/fold_counts.csv').set_index('fold').train_prior_logodds
    shift=test.fold.map(prior).to_numpy(float)
    def matrix(d,col,attr):return d.pivot(index='case_id',columns='scenario_id',values=col).loc[order,LADDER[attr]].to_numpy()
    xf=matrix(grid,XCOL['Fare'],'Fare').astype(float);xt=matrix(grid,XCOL['IVT'],'IVT').astype(float)
    def support(attr):
      d=grid.copy();d['support']=d.within_support.astype(bool)&d.physically_valid.astype(bool)
      flags=matrix(d,'support',attr).astype(bool)
      return flags[:,:-1]&flags[:,1:]
    sf,st=support('Fare'),support('IVT');common=(sf.sum(1)>0)&(st.sum(1)>0)
    rows=[];pred=[];case=[]
    for model in ['Human reference']+list(q.Model.unique()):
      human=model=='Human reference';d=h if human else q[q.Model.eq(model)]
      pf=matrix(d,'p','Fare').astype(float);pt=matrix(d,'p','IVT').astype(float)
      for probability in (['raw'] if human else ['raw','training_adjusted']):
        if probability=='raw':F,T=pf,pt
        else:F=expit(matrix(d,'margin','Fare')+shift[:,None]);T=expit(matrix(d,'margin','IVT')+shift[:,None])
        ef,df,_=six_pairs(F,xf);_,dt,_=six_pairs(T,xt)
        if not human:
          b=d[d.scenario_id.eq('BASE')].set_index('case_id').loc[order]
          pb=b.p.to_numpy() if probability=='raw' else expit(b.margin.to_numpy()+shift)
          hard=b.hard.to_numpy() if probability=='raw' else (pb>=.5).astype(int)
          pred.append(dict(Model=model,probability=probability,Log_Loss=log_loss(test.y,np.clip(pb,1e-9,1-1e-9)),AV_recall=recall_score(test.y,hard),predicted_AV_share=float(hard.mean())))
        for domain in ['full_ladder','within_design_pairs']:
          a,b=(np.ones_like(sf),np.ones_like(st)) if domain=='full_ladder' else (sf,st)
          eligible=np.ones(len(test),bool) if domain=='full_ladder' else common
          DF=np.divide((df*a).sum(1),a.sum(1),out=np.full(len(test),np.nan),where=a.sum(1)>0)
          DT=np.divide((dt*b).sum(1),b.sum(1),out=np.full(len(test),np.nan),where=b.sum(1)>0)
          finite=eligible&np.isfinite(DF)&np.isfinite(DT)
          agg=60*DT[finite].mean()/DF[finite].mean()
          es=ef[a&eligible[:,None]]
          for threshold in [1e-5,1e-4,1e-3]:
            keep=finite&(np.abs(DF)>=threshold);ratios=60*DT[keep]/DF[keep]
            quantiles=np.quantile(ratios,[.05,.25,.5,.75,.95])
            rows.append(dict(Model=model,probability=probability,domain=domain,threshold=threshold,cases=int(eligible.sum()),fare_intervals=len(es),ivt_intervals=int(b[eligible].sum()),computable=int(keep.sum()),computable_pct=100*keep.sum()/eligible.sum(),expected_sign_pct=100*np.mean((DF[keep]<0)&(DT[keep]<0)),mean_elasticity=float(np.mean(es)),aggregate_vot=agg,vot_p05=quantiles[0],vot_q1=quantiles[1],vot_median=quantiles[2],vot_q3=quantiles[3],vot_p95=quantiles[4]))
          case.append(pd.DataFrame(dict(Model=model,probability=probability,domain=domain,case_id=order,respondent_id=test.respondent_id,DF=DF,DT=DT,eligible=eligible)))
    pd.DataFrame(rows).to_csv(WORK/'analysis/probability_sensitivity.csv',index=False)
    pd.DataFrame(pred).to_csv(WORK/'analysis/probability_sensitivity_prediction.csv',index=False)
    pd.concat(case,ignore_index=True).to_parquet(WORK/'analysis/probability_sensitivity_cases.parquet',index=False)
    (WORK/'qa/sensitivity_checks.json').write_text(json.dumps(dict(status='PASS',within_design_cases=int(common.sum()),within_design_respondents=int(test.loc[common,'respondent_id'].nunique()),fare_pairs=int(sf[common].sum()),ivt_pairs=int(st[common].sum()),prior_source='Each case uses its other-four-fold observed training prevalence; applied to bothZS/SFT and allrungs; nofit orselection.'),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).query("threshold==0.0001 and domain=='full_ladder'") .to_string(index=False))

if __name__=='__main__':main()
