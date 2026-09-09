"""Record, without hiding, the concurrent approved Chapter 2 source update."""
from pathlib import Path
from zipfile import ZipFile
import hashlib,json,shutil
W=Path(__file__).resolve().parents[1];Q=W/'qa';P=W.parent/'paper_en_partc';NEW=W.parent/'paper_en_partc_cv5'
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    initial=json.loads((Q/'original_hashes.json').read_text())
    current={f:sha((W.parent/f).read_bytes()) for f in initial}
    changed=[f for f in initial if initial[f]!=current[f]]
    evidence=W.parent/'CHAPTER2_APPROVED_20260908'
    package=json.loads((evidence/'package_checks.json').read_text())
    zpath=Path(package['zip']);assert sha(zpath.read_bytes())==package['sha256']
    source_names=['references.bib','sections_en/02_literature.tex','sections_en/04_results.tex']
    with ZipFile(zpath) as z:
        for name in source_names[:2]:assert z.read(name)==(P/name).read_bytes()
        base=W/'source_base'
        if not base.exists():
            base.mkdir();z.extractall(base)
        for name in z.namelist():assert z.read(name)==(base/name).read_bytes()
    later=W.parent/'SECTIONS41_42_APPROVED_20260908/package_checks.json'
    later_package=json.loads(later.read_text()) if later.exists() else None
    if later_package:
        with ZipFile(later_package['zip']) as z:
            for name in source_names:assert z.read(name)==(P/name).read_bytes()
    allowed={str(Path('paper_en_partc')/n) for n in source_names}
    allowed|={str(Path('paper_en_partc')/(stem+'.'+ext)) for stem in ['main_en','main_en_anonymous'] for ext in ['aux','bbl','blg','log','pdf']}
    assert set(changed).issubset(allowed),set(changed)-allowed
    assert (evidence/'clean_overleaf/main_en.pdf').exists() or (evidence/'main_en_text.txt').exists()
    report={'reason':'Concurrent independently completed author-approved Chapter 2 revision D369 and Sections4.1/4.2 revision D370; not mutations made by the CV5 task',
            'initial_hash_manifest_retained':'original_hashes.json','accepted_source_evidence':str(evidence),
            'verified_zip':str(zpath),'verified_zip_sha256':package['sha256'],'externally_changed_files':changed,
            'source_changes':['Approved Literature Review','Added Liu dual-agent bibliography entry','Concurrent Results editing and its PDF builds'],
            'separate_thread':'01a0773c-db3f-7281-8744-ee3cce396965','later_completed_package':later_package,
            'cv5_source_base':'Immutable completed D369 ZIP extracted to source_base; CV5 Results are separately revised for the new numerical evidence',
            'new_preservation_baseline':'original_hashes_current.json','unchanged_outside_separate_revision':len(initial)-len(changed)}
    (Q/'source_refresh.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (Q/'original_hashes_current.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
    snapshots=Q/'initial_source_snapshot';snapshots.mkdir(exist_ok=True)
    for name in ['references.bib','sections_en/02_literature.tex']:
        target=snapshots/Path(name).name
        if not target.exists():shutil.copy2(NEW/name,target)
        shutil.copy2(P/name,NEW/name)
    print(f'Recorded {len(changed)} concurrent changed artifacts; current original sources preserved and latest Chapter2 copied.')
if __name__=='__main__':main()
