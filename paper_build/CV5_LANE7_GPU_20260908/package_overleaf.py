"""Assemble an isolated portable CV5 manuscript and verify a clean build."""
from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,subprocess
import fitz
W=Path(__file__).resolve().parents[1];Q=W/'qa';P=W.parent/'paper_en_partc_cv5'
OLD=W.parent/'output/overleaf/TRC_English_Overleaf_20260908_Chapter2Revised.zip'
NEW=OLD.with_name('TRC_English_Overleaf_CV5_Lane7GPU_20260908.zip');CLEAN=Q/'clean_overleaf_lane7'
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    assert not CLEAN.exists(),'Use a fresh clean-build directory; do not delete another workspace.'
    old_hash=sha(OLD.read_bytes())
    with zipfile.ZipFile(OLD) as z:payload={n:z.read(n) for n in z.namelist()}
    for n in list(payload):
        p=P/n
        if p.is_file():payload[n]=p.read_bytes()
    pages=json.loads((Q/'build_checks.json').read_text())['builds']['main_en']['pages']
    payload['README_Overleaf.md']=f'''# TRC English manuscript: separate CV5 version

This package is the retrospective respondent-level five-fold extension, assembled
2026-09-08. It preserves Figures 1–7, Tables 1–8, all four Section 4.2 subsections,
both main-text subgroup boxplots, and the three-part selective appendix.
The earlier fixed-split manuscript is preserved separately.

## Compile

- Compiler: pdfLaTeX. Bibliography: BibTeX.
- Named document: main_en.tex. Anonymous document: main_en_anonymous.tex.
- Keep bundled CAS files, fonts and licenses in their supplied locations.
- Verified local result: {pages} pages; TeX distributions may vary in pagination.

## Evidence and editing

- Common candidate evaluation: 651 respondents, 3,468 cases, five assigned folds.
- BASE and changed conditions use the same held-out respondent's assigned model.
- Table 3 reports fold mean ± sample SD; pooled metrics are a separate diagnostic.
- Full-survey human references use 2,169 respondents and 11,584 retained choices.
  These references are descriptive and overlap with the evaluated cohort.
- Main chapters are in sections_en/. Abstract and title are in main_en.tex.
- Figure 1 is the unchanged author JPEG; retain the supplied trim options.
- Numerical figures are in figures_trimmed/. Display-boundary markers denote
  offscale values, not data deletion. Tables remain editable LaTeX.
- Appendix A documents implementation; B covers references and service ladders;
  C reports the focused comparison under the same numerical calculation.

No new Qwen inference or training was conducted for manuscript preparation.
FFNN and DNN use the author-selected lane-7 configurations, refitted on GPU with
seed 42 in each outer fold. Platt calibration and AV-F1 thresholds use three-fold
inner OOF predictions only. The other four conventional candidates retain their
completed CV5 fits. Fold metadata is excluded from predictors. Numerical and interpretive statements
were updated for the new evidence, including unresolved directional, subgroup
and economic-representation differences.

The project-local CV5_LANE7_GPU_20260908 directory retains reproducible analysis code,
input hashes, fitted models, validation reports and numerical_changes.csv.
Raw respondent records, model scores, training logs, older drafts and build files
are excluded from this submission ZIP.
'''.encode('utf-8')
    manifest={'version':'CV5_Lane7GPU_20260908','revision':'Retrospective five-fold manuscript with GPU lane7 calibrated FFNN/DNN','files':[]}
    for n,b in payload.items():
        if n!='MANIFEST.json':manifest['files'].append({'path':n,'bytes':len(b),'sha256':sha(b)})
    payload['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    with zipfile.ZipFile(NEW,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,b in payload.items():z.writestr(n,b)
    with zipfile.ZipFile(NEW) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in z.namelist())
        for item in manifest['files']:assert sha(z.read(item['path']))==item['sha256']
        CLEAN.mkdir();z.extractall(CLEAN)
    assert sha(OLD.read_bytes())==old_hash
    report={'status':'PASS','zip':str(NEW),'bytes':NEW.stat().st_size,'sha256':sha(NEW.read_bytes()),'entries':len(payload),'previous_zip_preserved':True,'clean_builds':{}}
    with (Q/'clean_build.log').open('w',encoding='utf-8') as log:
        def run(args):
            r=subprocess.run(args,cwd=CLEAN,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            log.write(r.stdout.decode(errors='replace'));log.flush();assert r.returncode==0,args
        for stem in ['main_en','main_en_anonymous']:
            cmd=['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'];run(cmd);run(['bibtex',stem])
            for _ in range(5):
                run(cmd);t=(CLEAN/(stem+'.log')).read_text(errors='replace')
                if not any(s in t for s in ['Label(s) may have changed','There were undefined','Rerun to get cross-references right']):break
            assert not any(s in t for s in ['Overfull','There were undefined','Missing character:','Label(s) may have changed'])
            with fitz.open(CLEAN/(stem+'.pdf')) as a,fitz.open(P/(stem+'.pdf')) as b:
                assert len(a)==len(b)==pages
                assert all(a[i].get_text()==b[i].get_text() and a[i].get_pixmap().samples==b[i].get_pixmap().samples for i in range(pages))
            report['clean_builds'][stem]={'pages':pages,'all_page_text_and_pixel_parity':True}
            print(stem,'clean-build PASS',flush=True)
    (Q/'package_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(str(NEW),flush=True)
if __name__=='__main__':main()
