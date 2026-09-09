"""Record the completed visual review and required project validator closure."""
from pathlib import Path
import json
import shutil

W = Path(__file__).resolve().parents[1]
Q = W / 'qa'
ROOT = Path('C:/SP_LLM/TRC')

source = Path('C:/SP_LLM/reports/phase_status/validation_report.md')
report = source.read_text(encoding='utf-8')
assert 'Overall status: PASS' in report
assert 'Checks passed: 32 / 32' in report
shutil.copy2(source, Q / 'root_validation_report.md')

visual = {
    'status': 'PASS',
    'named_pages_reviewed': 25,
    'method': 'All-page contact sheets and full-size inspection of dense result and appendix pages',
    'anonymous_review': 'First page visually inspected; remaining body verified against named PDF by all-page text/pixel checks',
    'figures': 'All six regenerated result figures reviewed at full size',
    'resolved_items': [
        'Figure 3 legend placement and percentage denominator note',
        'Appendix A.2 table font size',
        'SFT objective cross-reference to Appendix A.3',
        'Section 4.3 table and figure placement',
    ],
    'evidence': ['build_checks.json', 'package_checks.json', 'renders/'],
}
(Q / 'visual_review.json').write_text(json.dumps(visual, indent=2), encoding='utf-8')

record_path = Q / 'project_record.json'
record = json.loads(record_path.read_text(encoding='utf-8'))
record.update(root_validator='PASS', root_validator_checks='32/32',
              root_validator_scope='Existing project P0 data/output gate; separate from CV5 paper-specific validation')
record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')

marker = 'D371 CV5 required validator closure'
entry = f'''\n\n### 2026-09-08 - {marker}

- Executed `python scripts/99_validate_outputs.py` from `C:/SP_LLM` with the project Python runtime: PASS, 32/32 checks.
- Saved the report in `Real_exp/pdf/CV5_PAPER_20260908/qa/root_validation_report.md`. This verifies the existing P0 data/output gate; it supplements the 465 independent CV5 checks and does not replace the manuscript-specific numerical and visual review.
- Final named and anonymous PDFs each contain 25 pages. All pages and result figures were reviewed; the standalone Overleaf ZIP reproduces both PDFs with every-page text and pixel parity.
- D371 implementation and validation are complete. Next task: author review of the separate CV5 Tables 3–8 and the revised, qualified behavioral interpretation.
'''
for name in ['DECISIONS.md', 'RUN_LOG.md', 'PHASE_STATUS.md']:
    path = ROOT / name
    # Historical logs contain mixed encodings; inspect the ASCII marker without
    # decoding or rewriting existing records.
    if marker.encode('ascii') not in path.read_bytes():
        with path.open('a', encoding='utf-8') as stream:
            stream.write(entry)
print(json.dumps(record, indent=2))
