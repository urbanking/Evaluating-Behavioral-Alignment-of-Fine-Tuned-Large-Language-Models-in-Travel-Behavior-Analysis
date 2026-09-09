"""Preserve the approved calculation helpers and plot layout as local source snapshots."""
from pathlib import Path
import hashlib, json, shutil

Q = Path(__file__).resolve().parents[1]
R = Q.parents[1]
OUT = Q / 'figures/source_snapshots'
OUT.mkdir(exist_ok=True)
sources = [R / 'tools/section4_results_case_support.py', R / 'tools/section4_results_final_support.py',
           R.parent / 'scripts/six_pair_operator.py',
           R / 'pdf/FIGURE02_LEGIBILITY_EN_20260907/plot_figure02_cv5_style_snapshot.py']
rows = []
for source in sources:
    target = OUT / source.name
    if not target.exists(): shutil.copy2(source, target)
    assert source.read_bytes() == target.read_bytes(), 'Frozen layout helper changed: ' + str(source)
    rows.append({'source': str(source), 'snapshot': str(target.relative_to(Q)),
                 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(OUT / 'manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print('Four approved helper/style source snapshots frozen')
