"""Describe the final calibrated neural candidates and their measured outputs."""
from pathlib import Path
import json
import pandas as pd
W=Path(__file__).resolve().parents[1];P=W.parent/'paper_en_partc_cv5';A=W/'analysis'

def replace_paragraph(s,prefix,new):
    a=s.index(prefix);b=s.find('\n\n',a)
    return s[:a]+new+s[b if b>=0 else len(s):]

def main():
    m=pd.read_csv(A/'prediction_mean_sd.csv').set_index('Model')
    v=pd.read_csv(A/'direct_vot_summary.csv').set_index('Model')
    f=pd.read_csv(A/'direct_fare_summary.csv').set_index('Model')
    c=pd.read_csv(A/'correspondence.csv').set_index('Model')
    manifests={name:[json.loads((W/'nn_lane7/manifests'/f'f{i}_{name}.json').read_text()) for i in range(1,6)] for name in ['ffnn','dnn']}
    path=P/'sections_en/03_method.tex';s=path.read_text(encoding='utf-8')
    prefix='Conventional models receive the inputs'
    a=s.index(prefix);b=s.index('\n\n',a)
    extra=r'''The FFNN and DNN use a previously selected configuration with focal loss, followed by Platt probability calibration. Within each outer training sample, three-fold respondent-grouped out-of-fold neural predictions are used to fit a nonnegative-slope Platt transform and select a threshold maximizing AV-class F1. The final network, calibrator, and threshold are frozen before scoring the outer fold. The same calibrated probabilities supply Log Loss, Direct responses, and all BASE/CF analyses, while the same fitted threshold supplies generated choices. Thus, neural probability calibration and the decision threshold are parts of the evaluated system. The configuration was selected retrospectively from earlier experiments; the current outer-fold outcomes are not used to refit its calibration or threshold.'''
    s=s[:b]+'\n\n'+extra+s[b:];path.write_text(s,encoding='utf-8')

    path=P/'sections_en/appendix.tex';s=path.read_text(encoding='utf-8')
    s=s.replace(r'\caption{Conventional-model settings selected within each outer training sample.}',r'\caption{Conventional-model configurations and training settings.}')
    s=replace_paragraph(s,'For each model and outer fold,',r'''For Pooled logit, SVM, Random Forest, and XGBoost, three-fold respondent-grouped inner cross-validation selects hyperparameters by negative Log Loss using the four outer training folds. The previously fixed class-treatment rules are then applied. Their hard choices use $p\geq0.5$. The neural architectures and focal-loss settings in Table~\ref{tab:app-selected} are fixed from the earlier AV-identification comparison, rather than reselected on the current outer-fold outcomes. All candidates use seed 42; the earlier single-split averages over three seeds are not included in the five-fold performance table.''')
    where=s.index('\n\n',s.index('For Pooled logit, SVM'))
    thresholds={k:', '.join(f"{z['threshold']:.3f}" for z in vals) for k,vals in manifests.items()}
    new=rf'''For each neural candidate and outer fold, three-fold grouped inner predictions are obtained with preprocessing fitted only on the corresponding inner training respondents. Each network uses a further grouped 15\% split for PR-AUC early stopping. A logistic transform $p=\sigma[a\,\mathrm{{logit}}(p_{{raw}})+b]$, constrained to $a\geq0$, is fitted to the pooled inner out-of-fold predictions. Raw probabilities are clipped to $[10^{{-6}},1-10^{{-6}}]$ only to evaluate this logit. A threshold maximizing AV-class F1 is selected on the calibrated inner predictions over 0.02--0.98 in steps of 0.005, choosing the smallest threshold in a tie. The final network is trained within the outer training sample with the same grouped stopping rule. Its frozen calibrator and threshold are applied to BASE and every CF condition without refitting. Thresholds for folds 1--5 are {thresholds['ffnn']} for FFNN and {thresholds['dnn']} for DNN. Training uses PyTorch on an NVIDIA RTX 3080. All splits, fitted preprocessing, weights, calibration coefficients, and thresholds are retained in the reproducibility files.'''
    s=s[:where]+'\n\n'+new+s[where:]
    s=s.replace('\\FloatBarrier\n\\clearpage\n\\subsection{Qwen prompt and answer extraction}',
                '\\FloatBarrier\n\\subsection{Qwen prompt and answer extraction}')
    s=s.replace('\\clearpage\n\\subsection{Fine-tuning configuration and objective}',
                '\\FloatBarrier\n\\subsection{Fine-tuning configuration and objective}')
    path.write_text(s,encoding='utf-8')

    path=P/'sections_en/04_results.tex';s=path.read_text(encoding='utf-8')
    # Add the actual neural result without substituting the earlier single-split means.
    pos=s.index('\n\nFor Qwen, SFT increases')
    paragraph=rf'''The calibrated FFNN and DNN have mean Accuracy {m.loc['FFNN','Accuracy_mean']:.3f} and {m.loc['DNN','Accuracy_mean']:.3f}, Macro-F1 {m.loc['FFNN','Macro_F1_mean']:.3f} and {m.loc['DNN','Macro_F1_mean']:.3f}, and Log Loss {m.loc['FFNN','Log_Loss_mean']:.3f} and {m.loc['DNN','Log_Loss_mean']:.3f}, respectively. Their probability losses are lower than those of both Qwen SFT systems, while their Macro-F1 values are lower. The neural classification metrics use training-fitted AV-identification thresholds; the differences therefore concern each complete prediction procedure, including its loss, calibration, and decision rule.'''
    s=s[:pos]+'\n\n'+paragraph+s[pos:]
    key='These neural distributions use the same calibrated probability functions as the prediction evaluation.'
    s=s.replace(key,key+rf" Their mean Direct VOT ratios are {v.loc['FFNN','Population Direct VOT-eq.']:.3f} and {v.loc['DNN','Population Direct VOT-eq.']:.3f} kKRW/h, respectively. Despite predominantly negative fare responses, most computable neural cases have a positive IVT response, producing a negative time--cost ratio.")
    prefix='The conventional models show why these comparisons are separate.'
    s=s.replace(prefix,prefix+rf" FFNN and DNN have BASE-generated-choice VOT estimates of {c.loc['FFNN','GC-MNL VOT']:.3f} and {c.loc['DNN','GC-MNL VOT']:.3f} kKRW/h, whereas their aggregate Direct ratios are negative ({v.loc['FFNN','Population Direct VOT-eq.']:.3f} and {v.loc['DNN','Population Direct VOT-eq.']:.3f}).")
    note='SD describes fold differences, not repeated-seed uncertainty.'
    s=s.replace(note,note+' FFNN/DNN use training-fitted Platt probabilities and AV-F1 thresholds consistently across all analyses.')
    heading='\\subsubsection{Subgroup fare sensitivity and Direct VOT}'
    s=s.replace(heading,'\\FloatBarrier\n\n'+heading)
    path.write_text(s,encoding='utf-8')

    path=P/'sections_en/05_06_discussion_conclusion.tex';s=path.read_text(encoding='utf-8')
    # Keep the existing four Discussion subsections and nine Conclusion paragraphs.
    pos=s.index('\n\nApplying the common discrete-choice specification')
    neural=rf'''The calibrated neural comparisons reinforce the need to evaluate probability prediction and service responses separately. FFNN and DNN achieve lower Log Loss than both Qwen SFT systems, yet their Direct VOT ratios are predominantly negative because IVT often increases the predicted adoption probability. Better probability scores therefore do not establish economically expected attribute responses. This contrast does not isolate an architectural effect: the neural candidates combine focal-loss training, calibration, and an adoption-oriented threshold, while Qwen uses its stated answer-probability rule. Each result describes the complete fitted system under its documented procedure.'''
    s=s[:pos]+'\n\n'+neural+s[pos:]
    sentence='The five-fold analysis was assembled after the earlier fixed-split analysis and is a retrospective extension, not a preregistered independent replication.'
    s=s.replace(sentence,sentence+' The neural configuration was also chosen from the earlier comparison before this rerun; its inner calibration and threshold fitting exclude the assigned outer respondents, but this does not remove the retrospective nature of the configuration choice.')
    s=s.replace('The conventional candidates were refitted using respondent-grouped inner tuning within each outer training sample, with fold metadata excluded from predictors.',
                'The four non-neural conventional candidates use respondent-grouped inner tuning within each outer training sample. The neural candidates instead retain the selected architecture and focal loss, with inner-only early stopping, probability calibration, and threshold selection. Fold metadata is excluded from all predictors.')
    path.write_text(s,encoding='utf-8')
    print('Lane7 methods, settings, neural results, and qualified discussion updated.')
if __name__=='__main__':main()
