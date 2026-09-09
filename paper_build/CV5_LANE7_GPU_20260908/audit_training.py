"""Check final-adapter manifests and supervised train/eval case separation."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
TRC=Path('C:/SP_LLM/TRC');WORK=Path(__file__).resolve().parents[1]
def main():
    counts=pd.read_csv(WORK/'qa/fold_counts.csv').set_index('fold')
    pool=pd.read_parquet(WORK/'full_sample_index.parquet').set_index('case_id')
    rows=[]
    for family in ['qwen_mid','qwen']:
      for fold in range(1,6):
        root=TRC/f'artifacts/models/llm/{family}/HUMAN/retro/seed42_aaai_composite_aaai_la1_reason_cv{fold}'
        p=root/'run_manifest.json';d=json.loads(p.read_text(encoding='utf-8-sig'))
        src=TRC/f'data/llm/CV5_f{fold}_train.jsonl'
        cases=[json.loads(line) for line in src.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        ids=[str(x['case_id']) for x in cases]
        train=pool.loc[ids]
        assert len(set(ids))==len(ids)==d['n_train']==counts.loc[fold,'train_cases']
        assert train.fold.ne(fold).all() and set(ids)==set(pool[pool.fold.ne(fold)].index)
        assert all(int(x['label'])==int(pool.loc[str(x['case_id']),'y']) for x in cases)
        assert d['epochs']==3 and d['seed']==42 and d['init_adapter'] is None
        assert d['optimizer_steps']==3*int(np.ceil(len(ids)/16))
        log=(TRC/f'logs/cv5_2b_f{fold}_r.out' if family=='qwen_mid' else TRC/f'Real_exp/pdf/QWEN9B_CV5_REMOTE_20260907/logs/cv5one_qwen_f{fold}_r.out').read_text(encoding='utf-8',errors='replace')
        objective=next((s.strip() for s in log.splitlines() if '[objective]' in s),'')
        assert all(s in objective for s in ['wbce=0.65','focal=0.25','brier=0.10','binary=0.90','format=0.10','logit_adjust_tau=1.00'])
        rows.append(dict(Model='Qwen 2B SFT' if family=='qwen_mid' else 'Qwen 9B SFT',fold=fold,train_cases=len(ids),train_respondents=train.respondent_id.nunique(),train_AV=int(train.y.sum()),optimizer_steps=d['optimizer_steps'],epochs=3,prior_logodds=float(np.log(train.y.mean()/(1-train.y.mean()))),objective_log=objective,manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),train_file_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),adapter_bytes=(root/'adapter_model.safetensors').stat().st_size,torch=d['environment']['torch'],transformers=d['environment']['transformers']))
    pd.DataFrame(rows).to_csv(WORK/'qa/qwen_training_audit.csv',index=False)
    print(pd.DataFrame(rows)[['Model','fold','train_cases','train_respondents','train_AV','optimizer_steps','prior_logodds']].to_string(index=False))

if __name__=='__main__':main()
