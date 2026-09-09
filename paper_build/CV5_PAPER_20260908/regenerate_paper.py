"""Reapply the CV5 manuscript edits from the preserved fixed-split sources.

Only the separately named CV5 target is written. Existing analysis and figure
outputs are required; this script does not train or infer with any candidate.
"""
from pathlib import Path
import json,shutil,subprocess,sys
W=Path(__file__).resolve().parents[1];P=W.parent/'paper_en_partc_cv5';OLD=W/'source_base'
def main():
    assert P.resolve()!=OLD.resolve() and P.name=='paper_en_partc_cv5'
    assert json.loads((W/'analysis/validation.json').read_text())['complete_all_models']
    assert json.loads((W/'figures/render_manifest.json').read_text())['status']=='PASS'
    for source in (OLD/'sections_en').glob('*.tex'):shutil.copy2(source,P/'sections_en'/source.name)
    for name in ['main_en.tex','main_en_anonymous.tex','references.bib']:shutil.copy2(OLD/name,P/name)
    for script in ['update_method.py','update_discussion.py','update_results_prose.py','update_tables.py','update_settings.py']:
        subprocess.run([sys.executable,str(W/'scripts'/script)],check=True,cwd=W.parent)
    print('Separate CV5 TeX regenerated; original source untouched.')
if __name__=='__main__':main()
