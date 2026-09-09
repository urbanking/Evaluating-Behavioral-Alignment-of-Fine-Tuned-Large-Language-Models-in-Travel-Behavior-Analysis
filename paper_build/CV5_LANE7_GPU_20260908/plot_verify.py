"""Asset-level structural and render checks for the six new scientific figures."""
from pathlib import Path
import hashlib, json
import fitz
from PIL import Image, ImageDraw

Q = Path(__file__).resolve().parents[1]
OUT = Q / 'figures'
P = Q.parent / 'paper_en_partc_cv5'
manifest = json.loads((OUT / 'render_manifest.json').read_text())
checks = []
for item in manifest['figures']:
    stem = item['stem']; f = OUT / (stem + '.pdf')
    with fitz.open(f) as doc:
        assert len(doc) == 1
        page = doc[0]; text = page.get_text()
        assert 'Qwen 2B' in text and 'Qwen 9B' in text
        assert 'Panel Logit' not in text
        assert not page.get_images(full=True), 'Scientific plots must remain vector artwork'
        if 'probability' not in stem:
            assert 'Pooled logit' in text
        if 'overall_direct_vot' in stem:
            assert 'Case-level distribution' in text and 'Cases (%)' in text
        if 'gc_mnl_direct_vot' in stem:
            assert '95% Fieller confidence set' in text
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(OUT / (stem + '_qa.png'))
        d = hashlib.sha256(f.read_bytes()).hexdigest()
        assert d == item['pdf_sha256']
        for folder in ['figures', 'figures_trimmed']:
            assert hashlib.sha256((P / folder / f.name).read_bytes()).hexdigest() == d
        checks.append({'stem': stem, 'pages': 1, 'vector_only': True, 'width_pt': page.rect.width,
                       'height_pt': page.rect.height, 'manuscript_asset_copies_equal': True})
sheet = Image.new('RGB', (1800, 1920), 'white'); draw = ImageDraw.Draw(sheet)
for i, item in enumerate(manifest['figures']):
    im = Image.open(OUT / (item['stem'] + '_qa.png')).convert('RGB')
    im.thumbnail((870, 595))
    x = (i % 2) * 900 + (900 - im.width) // 2
    y = (i // 2) * 640 + 35
    sheet.paste(im, (x, y)); draw.text(((i % 2) * 900 + 12, (i // 2) * 640 + 8), item['stem'], fill='black')
sheet.save(OUT / 'all_figures_contact.jpg', quality=94)
(OUT / 'figure_asset_qa.json').write_text(json.dumps({'status': 'PASS', 'figures': checks,
    'n_subgroup_boxes': manifest['n_subgroup_boxes'], 'subgroup_quantile_checks': manifest['subgroup_quantile_checks'],
    'axis_changes': manifest['axis_changes']}, indent=2), encoding='utf-8')
print('Six vector figure assets: structure, labels, copied hashes, and render checks PASS')
