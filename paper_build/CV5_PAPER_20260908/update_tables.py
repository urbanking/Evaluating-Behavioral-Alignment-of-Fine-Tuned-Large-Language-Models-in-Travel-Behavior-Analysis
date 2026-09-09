"""Replace numerical table bodies, preserving main/appendix table inventory."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
WORK=Path(__file__).resolve().parents[1];A=WORK/'analysis';P=WORK.parent/'paper_en_partc_cv5'
ORDER=['Panel Logit','SVM','Random Forest','XGBoost','FFNN','DNN','Qwen 2B ZS','Qwen 2B SFT','Qwen 9B ZS','Qwen 9B SFT']
QWEN=ORDER[-4:]
def name(x):return 'Pooled logit' if x=='Panel Logit' else x
def num(x,d=3):
    if pd.isna(x):return '--'
    if not np.isfinite(x):return r'$\infty$' if x>0 else r'$-\infty$'
    s=f'{float(x):.{d}f}'
    if float(s)==0:s=f'{0:.{d}f}'
    return '$'+s+'$' if s.startswith('-') else s
def row(c):return ' & '.join(c)+r' \\'
def replace_body(text,label,body):
    pos=text.index('\\label{'+label+'}');start=text.rfind('\\begin{table}',0,pos);end=text.index('\\end{table}',pos)+len('\\end{table}')
    block=text[start:end];a=block.index('\\midrule')+len('\\midrule');b=block.rfind('\\bottomrule')
    assert a<b
    return text[:start]+block[:a]+'\n'+body+'\n'+block[b:]+text[end:]
def note(text,label,new):
    pos=text.index('\\label{'+label+'}');end=text.index('\\end{table}',pos)
    a=text.index(r'{\footnotesize\raggedright Note:',pos,end);b=text.index(r'\par}',a,end)+len(r'\par}')
    return text[:a]+r'{\footnotesize\raggedright Note: '+new+r'\par}'+text[b:]
def main():
    m=pd.read_csv(A/'prediction_mean_sd.csv').set_index('Model')
    f=pd.read_csv(A/'direct_fare_summary.csv').set_index('Model')
    v=pd.read_csv(A/'direct_vot_summary.csv').set_index('Model')
    fits=pd.read_csv(A/'fit_audit.csv');base=fits[fits.projection.eq('BASE_MNL') & fits.fold.eq(0)].set_index('Model')
    co=pd.read_csv(A/'all_coefficients.csv');co=co[co.projection.eq('BASE_MNL') & co.fold.eq(0)].set_index(['Model','Term'])
    gc=pd.read_csv(A/'correspondence.csv').set_index('Model');cf=pd.read_csv(A/'cf_correspondence.csv').set_index('Model')
    assert all(set(ORDER).issubset(d.index) for d in [m,f,v,base,gc,cf]),'Full candidate analysis not ready'
    s=(P/'sections_en/04_results.tex').read_text(encoding='utf-8')
    body='\n'.join(row([name(model)]+[f"{m.loc[model,k+'_mean']:.3f} $\\pm$ {m.loc[model,k+'_sd']:.3f}" for k in ['Accuracy','Macro_F1','Log_Loss']]) for model in ORDER)
    s=replace_body(s,'tab:tab01',body)
    s=note(s,'tab:tab01',r'Mean $\pm$ sample SD across the five fold-specific portions of the common evaluation cohort (651 respondents; 3,468 cases). Each case uses a model trained without its respondent; ZS scores are partitioned by the same folds. BASE probabilities come from the same inference run as the condition-change analysis. SD describes fold differences, not repeated-seed uncertainty.')
    body=[]
    for model in ['Human reference']+ORDER:
      z=f.loc[model]
      body.append(row([name(model),f"{int(z['N']):,}"]+[num(z[k]) for k in ['Mean','SD','Median','Q1','Q3']]+[num(z['Negative (%)'],2),num(z['Absolute gap from Human mean'])]))
    s=replace_body(s,'tab:tab02','\n'.join(body))
    body=[]
    for model in ['Human same-CF reference']+ORDER:
      z=v.loc[model];label='Human (matched perturbations)' if model=='Human same-CF reference' else name(model)
      body.append(row([label,str(int(z['N cases']))]+[num(z[k],2) for k in ['Computable (%)','Fare-negative (%)','IVT-negative (%)','Joint conventional-sign (%)']]+[num(z['Population Direct VOT-eq.']),num(z['Median']),num(z['Q1'])+' -- '+num(z['Q3'])]))
    s=replace_body(s,'tab:tab03','\n'.join(body))
    def coef(model,term,d=3):
      z=co.loc[(model,term)];pv=z['p-value'];star='***' if pv<.01 else '**' if pv<.05 else '*' if pv<.10 else ''
      return num(z.Coefficient,d)+' ('+num(z['Cluster SE'],d)+')'+('$^{'+star+'}$' if star else '')
    blocks=[('Demographic',[('Age (years)','age',4),(r'Male (ref.\ female)','Male',3)]),('Household income',[('Income bracket score (1--4)','income_linear',3),('Income nonresponse','income_nonresponse',3)]),('Education',[(r'College (ref.\ high school or lower)','D2[2]',3),('Graduate school or higher','D2[3]',3)]),('Vehicle ownership',[(r'One vehicle (ref.\ zero)','D6[2]',3),('Two or more vehicles','D6[3]',3)]),('Incumbent mode',[(r'Car (ref.\ public transit)','current_mode_2026[Car]',3),('Personal mobility','current_mode_2026[PM]',3),('Walking','current_mode_2026[Walk]',3)])]
    body=[]
    for bi,(block,variables) in enumerate(blocks):
      if bi:body.append(r'\midrule')
      body.append(r'\multirow{'+str(len(variables))+r'}{*}{'+block+'}')
      for label,term,d in variables:body.append(row(['',label]+[coef(model,term,d) for model in ['H-MNL']+QWEN]))
    s=replace_body(s,'tab:tab05','\n'.join(body))
    s=note(s,'tab:tab05',r'These coefficients and the LOS coefficients in Table~\ref{tab:tab06} come from the same fitted models, controlling for incumbent-mode LOS and distance. Age is in years; income uses bracket scores 1--4, with score 1 plus a separate indicator for nonresponse. H-MNL uses all 11,584 observed survey choices; Qwen fits use 3,468 pooled out-of-fold BASE-generated choices. Parentheses contain respondent-clustered standard errors; $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    body=[]
    for model in ['H-MNL']+QWEN:
      z=base.loc[model];body.append(row([model]+[coef(model,a+'_AV',4) for a in ['fare','ivt','wait']]+[num(z.elasticity),num(z.vot)]))
    s=replace_body(s,'tab:tab06','\n'.join(body))
    hr=base.loc['H-MNL']
    s=note(s,'tab:tab06',r'All entries use the same 23-parameter fitted models as Table~\ref{tab:tab05}: continuous age, sex, linear income score and nonresponse indicator, education, vehicle ownership, incumbent-mode LOS and indicators, and distance controls; trip purpose is excluded. Parentheses contain respondent-clustered standard errors; $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. Fare is in kKRW, time in minutes, and VOT in kKRW/h. The 95\% H-MNL Fieller set is ['+num(hr.fieller_low)+', '+num(hr.fieller_high)+r']; H-MNL and both SFT GC-MNL sets are bounded but include zero. These intervals concern coefficient ratios, not the distribution of individual VOT or training-fold variability.')
    he=hr.elasticity;hv=hr.vot;hde=f.loc['Human reference','Mean'];hdv=v.loc['Human same-CF reference','Population Direct VOT-eq.']
    body=[]
    for panel in ['Fare','VOT']:
      if panel=='Fare':
        title=r'Panel A. Fare elasticity (Direct: mean interval elasticity)';hb,hd=he,hde
        keys=['GC-MNL Fare elasticity','Direct Fare elasticity','CF-MNL Fare elasticity']
      else:
        title=r'Panel B. Value of travel time (Direct: aggregate VOT; kKRW/h)';hb,hd=hv,hdv
        keys=['GC-MNL VOT','Direct VOT-eq.','CF-MNL VOT']
        body.append(r'\midrule')
      body.append(r'\multicolumn{8}{@{}l}{\itshape '+title+r'} \\')
      body.append(row(['Human reference',num(hb),'0.000','--',num(hd),'0.000',num(abs(hb-hd)),'--']))
      for model in ORDER:
        b=gc.loc[model,keys[0]];d=gc.loc[model,keys[1]];c=cf.loc[model,keys[2]]
        body.append(row([name(model)]+[num(x) for x in [b,abs(b-hb),c,d,abs(d-hd),abs(b-d),abs(c-d)]]))
    s=replace_body(s,'tab:tab07','\n'.join(body))
    s=note(s,'tab:tab07',r'The human rows display the references used to compute the gaps. GC-MNL estimates from pooled BASE-generated choices are compared with the full-survey H-MNL reference; Direct measures with its responses on the same service ladders. CF-MNL has no observed human counterfactual counterpart and is compared only with Direct. Gaps are absolute differences between scalar summaries, not distances between distributions. All candidate summaries use the same fold-assigned frozen models and seed 42 where applicable. The economic projections describe the pooled cross-fitted system; they do not establish the correspondence of every individual adapter. Table~\ref{tab:tab06} reports coefficient-ratio uncertainty.')
    (P/'sections_en/04_results.tex').write_text(s,encoding='utf-8')
    s=(P/'sections_en/appendix.tex').read_text(encoding='utf-8');d=pd.read_csv(A/'same_operator_gc_direct.csv').set_index('Model')
    body=[]
    for model in QWEN:
      z=d.loc[model]
      body.append(row([model,num(z.GC_MNL_mean_elasticity),num(z.Direct_mean_elasticity),num(z.GC_MNL_median_vot)+' ['+num(z.GC_MNL_q1_vot)+', '+num(z.GC_MNL_q3_vot)+']',num(z.Direct_median_vot)+' ['+num(z.Direct_q1_vot)+', '+num(z.Direct_q3_vot)+']']))
    s=replace_body(s,'tab:matched-diagnostics','\n'.join(body))
    gc_n=', '.join(f"{int(d.loc[model,'GC_MNL_vot_cases']):,}" for model in QWEN)
    direct_n=', '.join(f"{int(d.loc[model,'Direct_vot_cases']):,}" for model in QWEN)
    s=note(s,'tab:matched-diagnostics',r'Elasticities use all 20,808 adjacent fare intervals. VOT is in kKRW/h and retains signed case-level ratios with finite slopes and $|D_F|\geq10^{-4}$. In row order, GC-MNL VOT sample sizes are '+gc_n+'; Direct sample sizes are '+direct_n+r'. The denominators vary by model and representation. These numerically evaluated probability responses differ from the coefficient-based measures in Table~\ref{tab:tab07}.')
    extra='The pooled Qwen projections are estimable, but seven of the 20 fold-specific BASE projections exhibit quasi-complete separation in the income-nonresponse category and do not support finite maximum-likelihood coefficients. Their coefficients are not reported. The remaining 13 BASE and all 40 CF fold projections pass the fit checks. Accordingly, pooled correspondence does not establish that every individual adapter has a stable economic representation.'
    if extra not in s:s=s.rstrip()+'\n\n'+extra+'\n'
    (P/'sections_en/appendix.tex').write_text(s,encoding='utf-8')
    (WORK/'qa/table_sources.json').write_text(json.dumps({'table3':'prediction_mean_sd.csv','table4':'direct_fare_summary.csv','table5':'direct_vot_summary.csv','tables6_7':'same BASE_MNL fold0 rows in fit_audit.csv/all_coefficients.csv','table8':'correspondence.csv/cf_correspondence.csv plus human_reference','appendixC':'same_operator_gc_direct.csv','model_order':ORDER},indent=2),encoding='utf-8')
    print('All six Results tables and focused Appendix table updated.')

if __name__=='__main__':main()
