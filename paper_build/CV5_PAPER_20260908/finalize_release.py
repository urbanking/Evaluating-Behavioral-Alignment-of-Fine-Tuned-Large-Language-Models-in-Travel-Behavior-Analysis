"""Create a versioned result inventory and exact manuscript diff record."""
from pathlib import Path
import csv,difflib,hashlib,json
W=Path(__file__).resolve().parents[1];Q=W/'qa';P=W.parent/'paper_en_partc_cv5'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    for p in [Q/'independent_data_checks.json',Q/'package_checks.json',Q/'visual_review.json',W/'figures/render_manifest.json',W/'conventional/validation.json',W/'analysis/validation.json']:
        d=json.loads(p.read_text());assert d['status']=='PASS',p
    project_record=json.loads((Q/'project_record.json').read_text())
    assert project_record['root_validator']=='PASS'
    build=json.loads((Q/'build_checks.json').read_text())
    assert build['named_anonymous_body_equal']
    assert all(not d['warnings'] and not d['unknown_references'] for d in build['builds'].values())
    diffdir=Q/'manuscript_diffs';diffdir.mkdir(exist_ok=True)
    changed=[]
    for old in (W/'source_base').rglob('*'):
        if old.suffix not in ['.tex','.bib'] or not old.is_file():continue
        rel=old.relative_to(W/'source_base');new=P/rel
        if new.exists() and old.read_bytes()!=new.read_bytes():
            changed.append(str(rel))
            diff=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(keepends=True),new.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='D369/'+str(rel),tofile='CV5/'+str(rel)))
            (diffdir/(rel.name+'.diff')).write_text(diff,encoding='utf-8')
    rows=[]
    skip={'__pycache__','vendor','source_base','fit_cache','renders','clean_overleaf_cv5'}
    for p in sorted(W.rglob('*')):
        if not p.is_file() or any(x in skip for x in p.relative_to(W).parts):continue
        if p.name in ['result_inventory.csv','release_manifest.json'] or p.suffix in ['.log','.pyc']:continue
        rel=p.relative_to(W);rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha(p),'category':rel.parts[0] if len(rel.parts)>1 else 'release'})
    with (W/'result_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        out=csv.DictWriter(f,fieldnames=['path','bytes','sha256','category']);out.writeheader();out.writerows(rows)
    package=json.loads((Q/'package_checks.json').read_text())
    report={'status':'PASS','version':'CV5_20260908','source_revision':'D369 completed Chapter2 revision','paper_directory':str(P),
            'named_pdf':str(P/'main_en.pdf'),'anonymous_pdf':str(P/'main_en_anonymous.pdf'),'pages':build['builds']['main_en']['pages'],
            'main_figures':7,'main_tables':8,'section42_subsections':4,'appendix_tables':4,'conclusion_paragraphs':9,
            'candidate_cohort':[651,3468],'candidate_conditions':33,'candidate_models':10,'new_conventional_fits':30,
            'new_qwen_training_or_inference':False,'human_reference_sample':[2169,11584],'pooled_economic_fits':31,
            'bootstrap_replicates':200,'valid_qwen_fold_economic_fits':53,'attempted_qwen_fold_economic_fits':60,
            'source_hash_audit':'qa/source_refresh.json records concurrent separate original-paper edits; unaffected original files remain unchanged',
            'changed_tex_bib':changed,'inventory_files':len(rows),'overleaf_zip':package['zip'],'zip_sha256':package['sha256'],
            'required_root_validator':project_record['root_validator_checks']+' PASS','visual_review':'PASS'}
    (W/'release_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
