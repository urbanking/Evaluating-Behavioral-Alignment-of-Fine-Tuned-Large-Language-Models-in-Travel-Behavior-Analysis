"""Account for independently completed D372 without adopting its fixed-split text."""
from pathlib import Path
import hashlib,json,zipfile
W=Path(__file__).resolve().parents[1];P=W.parent;E=P/'CONCLUSION_BILINGUAL_20260908';Q=W/'qa'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
done=json.loads((E/'completion_checks.json').read_text());assert done['status']=='PASS' and done['decision']=='D372'
pack=json.loads((E/'package_checks.json').read_text());assert sha(pack['zip'])==pack['sha256']
with zipfile.ZipFile(pack['zip']) as z:
    rel='sections_en/05_06_discussion_conclusion.tex'
    assert z.read(rel)==(P/'paper_en_partc'/rel).read_bytes()
src=json.loads((E/'source_checks.json').read_text())
for lang in ['en','kr']:
    for stem,info in src['languages'][lang]['pdfs'].items():assert sha(P/f'paper_{lang}_partc'/f'{stem}.pdf')==info['sha256']
assert (E/'conclusion_kr.tex').read_text(encoding='utf-8').strip() in (P/'paper_kr_partc/sections_kr/05_06_discussion_conclusion.tex').read_text(encoding='utf-8')
old=json.loads((Q/'original_hashes.json').read_text());r=json.loads((Q/'source_refresh.json').read_text())
new=[]
for rel,h in old.items():
    if sha(P/rel)!=h and rel not in r['externally_changed_files']:
        assert rel.startswith(('paper_en_partc','paper_kr_partc'))
        assert Path(rel).name in ['05_06_discussion_conclusion.tex','main_kr.aux','main_kr.log','main_kr.pdf','main_kr_anonymous.aux','main_kr_anonymous.log','main_kr_anonymous.pdf'],rel
        new.append(rel)
r['externally_changed_files']=sorted(set(r['externally_changed_files']+new))
r['D372_external_conclusion']={'completed_evidence':str(E),'verified_package':pack,'additional_changed_files':new,
                             'note':'Fixed-split original English/Korean conclusion edit by another user task; CV5 text is maintained separately for its own results.'}
r['unchanged_outside_separate_revision']=len(old)-len(r['externally_changed_files'])
(Q/'source_refresh.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print('Verified external D372 changes:',len(new))
