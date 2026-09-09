"""Append completion records without overwriting other active task history."""
from pathlib import Path
import json,re
W=Path(__file__).resolve().parents[1];ROOT=Path('C:/SP_LLM/TRC');Q=W/'qa'
def main():
    assert json.loads((Q/'package_checks.json').read_text())['status']=='PASS'
    path=ROOT/'DECISIONS.md';text=path.read_text(encoding='utf-8')
    marker='Separate English CV5 manuscript with preserved main structure'
    if marker in text:
        print('Completion design record already present');return
    num=max(int(x) for x in re.findall(r'\bD(\d+)\b',text))+1;decision='D'+str(num)
    details=f'''\n\n### {decision} - {marker}

- Date: 2026-09-08. User explicitly approved implementation of a separate English five-fold manuscript in `Real_exp/pdf/paper_en_partc_cv5`, preserving main Figures1–7, Tables1–8, both subgroup boxplots, four Section4.2 subsections, and the selective AppendixA–C. No new Qwen training or inference was launched. Conclusion retains nine paragraphs.
- Candidate scope is the common 651respondents/3,468cases with 33 complete conditions per model. Qwen BASE scores come from the same completed condition-inference files; all four states have114,444case–condition keys. Each candidate case uses the model assigned to its respondent's held-out fold. The common cohort is not the full11,584case five-fold evaluation universe; the extension is retrospective after earlier fixed-split analysis.
- The user authorized full-survey descriptive human references: overall23parameters and separate subgroup29parameters each fit once on2,169respondents/11,584retained choices. This is a scoped exception to the usual held-out-fit restriction for descriptive human references only, not an independently validated prediction benchmark. Candidate preprocessing/tuning/fitting excludes each outer evaluation respondent. Overlapping human/evaluation bootstrap multiplicities are linked.
- Refit six conventional candidates across five folds, excluding fold/IDs/targets from the72 legitimate feature inputs, retaining historical training-only three-fold GroupKFold grids, fixed class-treatment overrides andseed42. Samefit suppliesBASE+CF. Thirty fits,686,664rows,coverage/leakage/source/model/hash checksPASS. SVMp=0.5 hard labels follow the existing p>=0.5 rule;254initial labels (5BASE) were normalized without changing probabilities. XGBoost import failures were recovered using isolated version3.0.2; historical version was not recorded. Runtime and clean reproduction are documented locally.
- Table3 reports fold mean and sampleSD; pooledOOF metrics remain separate. Numerical operators are unchanged: respondent-balanced curves, common waitsubset1,631cases/630respondents,20,808adjacent fare intervals,Figure6case means,case VOT thresholdabs(DF)>=1e-4,aggregate VOT ratio of all-finite mean slopes.
- All31 pooled23parameter economic fits pass; Tables6/7 share each fit and clustered covariance. Income1–4linear plus nonresponse, agecontinuous, sex/education/vehicles/incumbentLOS+mode/distance retained, purposeexcluded. Subgroup reference retains separate categorical income coding. Fold-specificQwen projections yield53valid/60attempts;7BASE fits have separation and are recorded, never averaged or forced. Linkedbootstrap200/200converged; no frozen-model training uncertainty is claimed.
- Results-dependent Methods/Abstract/Results/Discussion/Conclusion/Appendix updated. Qwen2BSFT has highest meanMacro-F1(.652),9BSFT.648; SVM highestAccuracy(.831),lowestLogLoss(.402). BothQwen meanfaregaps decrease, but2Bdispersion increases. VOTIQRs narrow while expected-sign shares decrease. 2BBASE-to-DirectVOTgap increases slightly;9Bgaps decrease. H-MNLIVTp=.098; VOT6.569kKRW/h withFieller[-1.198,15.051]. Bootstrap conclusions differ byendpoint/size and are revised accordingly.
- Regenerated six result figures preserve approved styles and raw signed distributions;42offscale boundary marks are disclosed, without statistical trimming. FixedFigure3legend overlap,TableA.2smalltype,andSFTobjective cross-reference. Float barriers keep4.3tables/figures with the appropriate result blocks. Both EnglishPDFs25pages; no overfull/undefined/missing-character warnings; anonymous body parityPASS. All pages visually reviewed. Independent data/operator/metric checks465PASS; figure/fit checksPASS.
- Source baseline is immutable completedD369Chapter2package. A separate user thread concurrently updated the original branch underD369/D370; its13changed artifacts are explicitly documented and85other originalEnglish/Koreanfiles retain initial hashes. This task does not overwrite the original branch; completed latestChapter2andreference are included inCV5. All previous ZIPs are preserved.
- Deliverables: `paper_en_partc_cv5/main_en.pdf`, `main_en_anonymous.pdf`; `output/overleaf/TRC_English_Overleaf_CV5_20260908.zip` (168entries; clean named/anonymous compilation has every-page text/pixel parity). `CV5_PAPER_20260908/numerical_changes.csv` contains2,857version comparisons with aggregation definitions; README, scripts, manifests, model artifacts and result inventory provide project-local reproduction. SubmissionZIP excludes raw records, score banks, modelweights and operational logs.
- Required rootP0validator is run after this entry and its result is appended separately. Next: author review of the separate CV5Tables3–8 and revised behavioral interpretation; no further Qwen experiment is required for this version.
'''
    with path.open('a',encoding='utf-8') as f:f.write(details)
    summary=f'''\n\n### 2026-09-08 - Separate English CV5 manuscript ({decision})

- Completed the authorized separate CV5 version, preserving7mainfigures/8maintables,subgroupplots,four4.2subsections,AppendixA–C,andnineConclusionparagraphs. Refit30conventionalmodels and31pooledeconomicrepresentations;200linkedbootstrapdraws. NoQwen training/inference.
- Commoncandidatecohort651/3,468;fullsurvey descriptivehumanreference2,169/11,584. Fold-only metadata removed frommodelinputs; foldmeanSDTable3 andpooledmetrics separated. Result-dependent text, uncertainty and allnumericalfigures/tables refreshed.
- BothPDFs25pages; independentdataQA465PASS,figure/fitQA andcleanOverleafrebuildPASS. ZIP: TRC_English_Overleaf_CV5_20260908.zip.2,857numericchanges andfullreprorecord inCV5_PAPER_20260908.
- Originalbranch untouched bythis task; concurrentD369/D370sourcechanges tracked separately. Next: author review of the separate CV5version. RootP0validatorresult follows.
'''
    with (ROOT/'RUN_LOG.md').open('a',encoding='utf-8') as f:f.write(summary)
    status=f'''\n\nTRC `Real_exp/pdf` separate English CV5 version, 2026-09-08: COMPLETE ({decision}). New paper_en_partc_cv5 uses common651respondents/3,468cases OOF predictions and full-survey descriptivehumanreferences. All30conventionalfits/31pooledeconomicfits complete; 200linkedbootstrapdraws; 465independentchecksPASS. Both PDFs25pages; Figure1–7/Table1–8/subgroupplots/four4.2subsections/AppendixA–C retained. StandaloneOverleafZIP clean buildsPASS. SourceconcurrencyD369/D370 documented without overwriting originals. Next: author review of CV5Tables3–8 and interpretation. Requiredrootvalidatorclosurefollows.\n'''
    with (ROOT/'PHASE_STATUS.md').open('a',encoding='utf-8') as f:f.write(status)
    (Q/'project_record.json').write_text(json.dumps({'decision':decision,'records_updated':['DECISIONS.md','RUN_LOG.md','PHASE_STATUS.md'],'root_validator':'pending'},indent=2))
    print(decision)
if __name__=='__main__':main()
