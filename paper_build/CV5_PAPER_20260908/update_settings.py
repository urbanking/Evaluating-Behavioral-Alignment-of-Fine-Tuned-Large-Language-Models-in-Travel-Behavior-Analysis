"""Write actual fold-specific conventional settings and figure display notes."""
from pathlib import Path
import json,re
import pandas as pd
W=Path(__file__).resolve().parents[1];P=W.parent/'paper_en_partc_cv5'
def main():
    groups={}
    for f in (W/'conventional/manifests').glob('f*.json'):
        if f.name.endswith('_features.json'):continue
        m=json.loads(f.read_text());groups.setdefault(m['model'],{})[m['fold']]=m['final_params']
    assert sum(map(len,groups.values()))==30
    def values(model,key):return [groups[model][i][key] for i in range(1,6)]
    def compressed(v,format=str):
        labels=[format(x) for x in v]
        return labels[0]+' (all folds)' if len(set(labels))==1 else ' / '.join(labels)
    svmC=compressed(values('svm','C'));svmG=compressed(values('svm','gamma'))
    rfL=compressed(values('random_forest','min_samples_leaf'));rfF=compressed(values('random_forest','max_features'))
    x=groups['xgboost'][1];assert all(z==x for z in groups['xgboost'].values())
    layer=lambda v:'--'.join(map(str,v))
    nn=[]
    for key,label in [('ffnn','FFNN'),('dnn','DNN')]:
        ls=compressed(values(key,'hidden_layer_sizes'),layer)
        al=compressed(values(key,'alpha'),lambda x:r'$10^{-4}$' if x==.0001 else r'$10^{-2}$')
        nn.append(label+' & Layers '+ls+'; '+r'$\alpha$: '+al+r'; early stopping, maximum 800 iterations & Fourfold positive-case replication \\')
    rows=[
        r'Pooled logit & Unpenalized lbfgs; reference-coded indicators; maximum 5,000 iterations & Balanced class weights \\',
        'SVM & RBF kernel; '+r'$C$: '+svmC+r'; $\gamma$: '+svmG+r'; internal probability scaling & Balanced class weights \\',
        'Random Forest & 500 trees; unrestricted depth; minimum leaf: '+rfL+'; feature fraction: '+rfF+r' & Balanced-subsample weights \\',
        f'XGBoost & 400 trees; depth {x["max_depth"]}; learning rate {x["learning_rate"]}; '+r'$\lambda=10$; row and column sampling 0.8 & Positive-class weight 4.31 \\',*nn]
    path=P/'sections_en/appendix.tex';s=path.read_text(encoding='utf-8')
    pos=s.index(r'\label{tab:app-selected}');a=s.index(r'\midrule',pos)+len(r'\midrule');b=s.index(r'\bottomrule',a)
    s=s[:a]+'\n'+'\n'.join(rows)+'\n'+s[b:]
    s=s.replace(r'\caption{Selected conventional-model settings.}',r'\caption{Conventional-model settings selected within each outer training sample.}')
    a=s.index(r'{\footnotesize\raggedright Note:',pos);b=s.index(r'\par}',a)+len(r'\par}')
    s=s[:a]+r'{\footnotesize\raggedright Note: Slash-separated settings follow folds 1--5. Neural early stopping uses a patience of 20 iterations. The outer-fold models all use seed 42. The pooled logit has no respondent random effects and is separate from H-MNL and the economic projection models. XGBoost uses version 3.0.2; the earlier experiment did not record its package version.\par}'+s[b:]
    target='The previously fixed class-treatment rule is then applied consistently.'
    replacement='The best inner-score configuration is then refitted with the previously fixed class-treatment rule, including positive-case replication for both neural candidates. Thus, the final class treatment is fixed by the historical comparison protocol rather than chosen anew from the outer evaluation results.'
    s=s.replace(target,replacement)
    s=s.replace('continuous age, income, incumbent mode, and distance, with age-group', 'continuous age, income-category indicators, incumbent mode, and distance, with age-group')
    heading=r'\subsection{Fine-tuning configuration and objective}'
    if r'\label{app:sft-objective}' not in s:s=s.replace(heading,heading+'\n'+r'\label{app:sft-objective}')
    pos=s.index(r'\label{tab:app-sft-full}');end=s.index(r'\end{table}',pos)
    s=s[:pos]+s[pos:end].replace(r'\scriptsize',r'\footnotesize')+s[end:]
    path.write_text(s,encoding='utf-8')
    pd.DataFrame([{'model':m,'fold':f,**z} for m,g in groups.items() for f,z in sorted(g.items())]).to_csv(W/'conventional/selected_settings_for_paper.csv',index=False)
    method=P/'sections_en/03_method.tex';text=method.read_text(encoding='utf-8')
    text=text.replace(r'Appendix~\ref{app:qwen-implementation} specifies the composite objective',r'Appendix~\ref{app:sft-objective} specifies the composite objective')
    phrase='The standard deviation describes differences across these folds; it is not a confidence interval or variation across repeated training seeds.'
    extra=r' For Log Loss only, probabilities are clipped to $[10^{-9},\,1-10^{-9}]$ for all candidates. Behavioral calculations use the original probabilities.'
    if extra not in text:text=text.replace(phrase,phrase+extra)
    method.write_text(text,encoding='utf-8')
    path=P/'sections_en/04_results.tex';s=path.read_text(encoding='utf-8')
    for label in ['fig:fig06','fig:fig10','fig:fig11']:
        pos=s.index('\\label{'+label+'}');end=s.index(r'\end{figure}',pos)
        a=s.index(r'\par}',pos,end)
        note=' Boundary triangles mark whiskers or estimates outside the display range; values are retained in the underlying summaries.'
        if note not in s[pos:end]:s=s[:a]+note+s[a:]
    label='fig:fig06';pos=s.index('\\label{'+label+'}');end=s.index(r'\end{figure}',pos)
    clause=' Computability is measured over all cases; expected-sign percentages are conditional on computable cases.'
    if clause not in s[pos:end]:
        a=s.index(r'\par}',pos,end);s=s[:a]+clause+s[a:]
    # Keep earlier tables/figures from interrupting the next 4.3 result paragraph.
    for title,label in [('Fare elasticity','sec:results-elasticity-correspondence'),('VOT','sec:results-vot-correspondence')]:
        target='\\subsubsection{'+title+'}\n\\label{'+label+'}'
        prefix='\\FloatBarrier\n\n'
        if prefix+target not in s:s=s.replace(target,prefix+target)
    path.write_text(s,encoding='utf-8')
    print('Actual settings and display-boundary notes updated')
if __name__=='__main__':main()
