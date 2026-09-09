"""Render the six CV5 result figures using the approved English layouts.

Only common-cohort CV5 candidate outputs and full-survey descriptive references
are accepted. The original fixed-split manuscript and its assets are never edited.
"""
from pathlib import Path
import hashlib, importlib.util, inspect, json, shutil, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

Q = Path(__file__).resolve().parents[1]
ROOT = Q.parents[1]
A = Q / 'analysis'
OUT = Q / 'figures'
PAPER = Q.parent / 'paper_en_partc_cv5'
sys.path.insert(0, str(ROOT.parent / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(OUT / 'source_snapshots'))
import section4_results_case_support as C

AXES = [('age_group', 'Age <50', 'Age 50+', 'A. Age'),
        ('sex_group', 'Male', 'Female', 'B. Sex'),
        ('framing_group', 'BASIC', 'RISK', 'C. Survey framing')]
MODELS = ['Human reference'] + C.MODEL_ORDER
DISPLAY = {'Panel Logit': 'Pooled logit'}
Y = [0, 1.55, 2.65, 3.75, 4.85, 5.95, 7.05, 8.8, 9.9, 11.35, 12.45]
MANIFEST = {'status': 'BUILDING', 'figures': [], 'axis_changes': []}
MANIFEST['offscale_display_marks'] = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expanded(old, values, pad=0.035):
    vals = np.asarray(values, float); vals = vals[np.isfinite(vals)]
    lo, hi = min(float(vals.min()), old[0]), max(float(vals.max()), old[1])
    if lo == old[0] and hi == old[1]:
        return old
    reserve = pad * max(hi - lo, 1)
    return (lo - reserve if lo < old[0] else old[0], hi + reserve if hi > old[1] else old[1])


def offscale(ax, value, y, color, stem, model, element, *, filled=True):
    """Mark every displayed whisker/estimate outside a fixed readable viewport."""
    lo, hi = ax.get_xlim(); value = float(value)
    if not np.isfinite(value) or lo <= value <= hi:
        return
    side = 'left' if value < lo else 'right'
    x = lo + .008 * (hi - lo) if side == 'left' else hi - .008 * (hi - lo)
    ax.plot(x, y, marker='<' if side == 'left' else '>', markersize=4.8,
            markerfacecolor=color if filled else 'white', markeredgecolor=color,
            linestyle='none', clip_on=False, zorder=9)
    MANIFEST['offscale_display_marks'].append({'stem': stem, 'model': model, 'element': element,
        'side': side, 'untrimmed_value': value, 'display_boundary': lo if side == 'left' else hi})


def save(fig, stem, *, bbox_pad=0.12, metadata=None):
    for ax in fig.axes:
        ticks = ax.get_yticklabels()
        if any(t.get_text() in DISPLAY for t in ticks):
            ax.set_yticks(ax.get_yticks(), [DISPLAY.get(t.get_text(), t.get_text()) for t in ticks])
        if ax.get_title() == 'A. Respondent distribution and aggregate VOT':
            ax.set_title('A. Case-level distribution and aggregate VOT')
        if ax.get_xlabel() == 'Respondents (%)':
            ax.set_xlabel('Cases (%)')
    for legend in fig.legends:
        for t in legend.get_texts():
            if t.get_text() == 'Direct-response distribution (IQR; whiskers p05-p95)' and 'elasticity' in stem:
                t.set_text('Direct case-mean distribution (IQR; whiskers p05-p95)')
            if t.get_text() == 'Human Fieller set':
                t.set_text('Survey VOT: 95% Fieller confidence set')
    pdf = OUT / (stem + '.pdf')
    fig.savefig(pdf, facecolor='white', bbox_inches='tight', pad_inches=bbox_pad)
    fig.savefig(OUT / (stem + '.png'), dpi=300, facecolor='white', bbox_inches='tight', pad_inches=bbox_pad)
    plt.close(fig)
    MANIFEST['figures'].append({'stem': stem, 'pdf_sha256': sha(pdf), **(metadata or {})})
    return OUT / (stem + '.png'), pdf


def subplot_tables_and_stats(interval, vot):
    meta = pd.read_csv(OUT / 'subgroup_respondent_meta.csv')
    hi = pd.read_parquet(OUT / 'subgroup_human_interval.parquet')
    hv = pd.read_parquet(OUT / 'subgroup_human_vot.parquet')
    e = pd.concat([hi, interval[interval.system.isin(C.MODEL_ORDER)]], ignore_index=True)
    v = pd.concat([hv, vot[vot.model.isin(C.MODEL_ORDER)]], ignore_index=True)
    for frame in [meta, e, v]:
        frame['respondent_id'] = frame.respondent_id.astype(str)
    assert e.groupby('system').size().eq(20808).all() and len(e.system.unique()) == 11
    assert v.groupby('model').size().eq(3468).all() and len(v.model.unique()) == 11
    et, vt = C.subgroup_tables(e, v, meta)
    et.to_csv(OUT / 'TableS_Figure04_subgroup_fare_elasticity.csv', index=False)
    vt.to_csv(OUT / 'TableS_Figure05_subgroup_direct_vot.csv', index=False)
    em = e.merge(meta, on='respondent_id', validate='many_to_one')
    vm = v.merge(meta, on='respondent_id', validate='many_to_one')
    stats = []
    for metric in ['fare_elasticity', 'direct_vot']:
        for axis, first, second, _ in AXES:
            for group in [first, second]:
                for model in MODELS:
                    if metric == 'fare_elasticity':
                        part = em[em[axis].eq(group) & em.system.eq(model)]
                        values = part.elasticity.to_numpy(float)
                    else:
                        key = C.HUMAN_V if model == C.HUMAN_E else model
                        part = vm[vm[axis].eq(group) & vm.model.eq(key)]
                        values = part.loc[part.raw_computable, 'raw_vot'].to_numpy(float)
                    values = values[np.isfinite(values)]
                    assert len(values) > 0
                    b = boxplot_stats(values, whis=(10, 90))[0]
                    r = dict(metric=metric, axis=axis, group=group, model=model, n=len(part), n_plotted=len(values))
                    r.update({k: float(b[k]) for k in ['q1', 'med', 'q3', 'whislo', 'whishi']})
                    r.update(p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)))
                    table = et if metric == 'fare_elasticity' else vt
                    row = table[table.Axis.eq(axis) & table.Group.eq(group) & table.Model.eq(model)].iloc[0]
                    np.testing.assert_allclose([r['q1'], r['med'], r['q3']], row[['Q1', 'Median', 'Q3']].to_numpy(float), atol=1e-12)
                    stats.append(r)
    assert len(stats) == 132
    (OUT / 'boxplot_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    contrasts = []
    for metric, table, measure in [('fare_elasticity', et, 'Mean'), ('direct_vot', vt, 'Aggregate')]:
        for axis, first, second, _ in AXES:
            for model in MODELS:
                rows = table[table.Axis.eq(axis) & table.Model.eq(model)].set_index('Group')
                contrasts.append({'metric': metric, 'axis': axis, 'model': model, 'first_group': first, 'second_group': second,
                                  'first_value': rows.loc[first, measure], 'second_value': rows.loc[second, measure],
                                  'second_minus_first': rows.loc[second, measure] - rows.loc[first, measure]})
    pd.DataFrame(contrasts).to_csv(OUT / 'subgroup_contrasts.csv', index=False)
    return stats


def subgroup_figures(stats):
    lookup = {(r['metric'], r['axis'], r['group'], r['model']): r for r in stats}
    blue, orange, fill = '#28668C', '#B7652E', '#E8B88F'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelcolor': '#30363A',
                         'axes.edgecolor': '#A8B0B6', 'text.color': '#252A2E', 'xtick.color': '#46515A',
                         'ytick.color': '#30363A', 'figure.facecolor': 'white', 'axes.facecolor': 'white', 'svg.fonttype': 'none'})
    for metric, stem, xlabel, oldlim in [
        ('fare_elasticity', 'Figure_03_fare_elasticity_subgroups', 'Direct fare elasticity', (-3.6, .9)),
        ('direct_vot', 'Figure_05_direct_vot_subgroups', 'Signed Direct VOT (kKRW/h)', (-35, 65))]:
        selected = [r for r in stats if r['metric'] == metric]
        xlim = oldlim
        fig, axes = plt.subplots(1, 3, figsize=(11.8, 8.2), sharey=True)
        fig.subplots_adjust(left=.16, right=.985, bottom=.18, top=.78, wspace=.16)
        count = 0
        for ax, (axis, first, second, panel) in zip(axes, AXES):
            ax.set_xlim(*xlim)
            ax.grid(False); ax.axvline(0, color='#818D96', linewidth=.9, zorder=0)
            for boundary in [.8, 7.95, 10.65]:
                ax.axhline(boundary, color='#C6CDD2', linewidth=.8, zorder=0)
            for model, y in zip(MODELS, Y):
                for j, (group, offset) in enumerate(zip([first, second], [-.24, .24])):
                    r = lookup[(metric, axis, group, model)]
                    b = {k: r[k] for k in ['q1', 'med', 'q3', 'whislo', 'whishi']}; b['fliers'] = []
                    color = blue if j == 0 else orange
                    result = ax.bxp([b], positions=[y + offset], widths=.36, orientation='horizontal',
                                    showfliers=False, patch_artist=True, manage_ticks=False,
                                    boxprops={'facecolor': 'white' if j == 0 else fill, 'edgecolor': color, 'linewidth': 1.15},
                                    whiskerprops={'color': color, 'linewidth': .9}, capprops={'color': color, 'linewidth': .9},
                                    medianprops={'color': '#22282D', 'linewidth': 1.5})
                    for box in result['boxes']: box.set_zorder(3)
                    for median in result['medians']: median.set_zorder(4)
                    offscale(ax, r['whislo'], y + offset, color, stem, model, f'{axis}:{group}:lower whisker', filled=j != 0)
                    offscale(ax, r['whishi'], y + offset, color, stem, model, f'{axis}:{group}:upper whisker', filled=j != 0)
                    count += 1
            ax.set_xlim(*xlim); ax.set_ylim(13.1, -.65)
            ax.set_xticks([-3, -2, -1, 0] if metric == 'fare_elasticity' else [-20, 0, 20, 40, 60])
            ax.set_yticks(Y, [DISPLAY.get(m, m) for m in MODELS])
            ax.tick_params(axis='y', length=0, pad=9, labelsize=13.2)
            ax.tick_params(axis='x', length=3, labelsize=11.7)
            ax.set_xlabel(xlabel, fontsize=12, labelpad=9)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.set_title(panel, fontsize=14, pad=49)
            ax.legend(handles=[Patch(facecolor='white', edgecolor=blue, linewidth=1.2, label=first),
                               Patch(facecolor=fill, edgecolor=orange, linewidth=1.2, label=second)],
                      loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=1, frameon=False, fontsize=12,
                      handlelength=1.3, labelspacing=.28, borderaxespad=0)
        assert count == 66
        marks = [r for r in MANIFEST['offscale_display_marks'] if r['stem'] == stem]
        save(fig, stem, bbox_pad=.04, metadata={'boxes': count, 'whiskers': [10, 90], 'xlim': xlim,
                                               'offscale_whiskers_marked': len(marks), 'data_filtered_for_display': False})


def response_figure():
    source = OUT / 'source_snapshots/plot_figure02_cv5_style_snapshot.py'
    spec = importlib.util.spec_from_file_location('approved_f2', source)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    data = pd.read_csv(A / 'response_curves.csv')
    selected = data[data.attribute.isin(mod.ATTRIBUTES) & data.system.isin(mod.SYSTEMS)].copy()
    assert len(selected) == 105 and not selected.duplicated(['system', 'attribute', 'rung']).any()
    assert selected.groupby('attribute').n_respondents.unique().apply(lambda x: len(x) == 1).all()
    assert selected[selected.attribute.eq('Wait')].n_respondents.eq(630).all()
    assert selected[~selected.attribute.eq('Wait')].n_respondents.eq(651).all()
    source_text = source.read_text(encoding='utf-8')
    snippet = source_text[source_text.index('    plt.rcParams.update({'):source_text.index('    installed = []')]
    snippet = '\n'.join(line[4:] if line.startswith('    ') else line for line in snippet.splitlines())
    ns = dict(plt=plt, np=np, pd=pd, selected=selected, OUT=OUT, STEM=mod.STEM, ATTRIBUTES=mod.ATTRIBUTES,
              SYSTEMS=mod.SYSTEMS, STYLE=mod.STYLE, X_LABEL=mod.X_LABEL)
    exec(compile(snippet, str(source), 'exec'), ns)
    assert len(ns['checked_series']) == 30
    MANIFEST['figures'].append({'stem': mod.STEM, 'source_style': str(source),
                               'exact_plotted_series_checks': len(ns['checked_series']),
                               'pdf_sha256': sha(OUT / (mod.STEM + '.pdf'))})


def overall_and_correspondence(vot, elasticity):
    vsum = pd.read_csv(A / 'direct_vot_summary.csv')
    corr = pd.read_csv(A / 'correspondence.csv')
    cf = pd.read_csv(A / 'cf_correspondence.csv')
    human = json.loads((A / 'human_reference.json').read_text())
    old_save = C._save
    def intercept(fig, out, stem):
        if stem == 'Figure_06_overall_direct_vot':
            # Exact existing viewport retained as a display constant, not old data.
            # The inherited lower-right legend obscures the final Qwen row.
            # Lower left is empty across the four Qwen rows in this final bank.
            fig.axes[1].legend(loc='lower left', frameon=False)
            limits = [-39.522239734406014, 100.0]
            ax = fig.axes[0]; ax.set_xlim(*limits)
            for yi, model in enumerate([C.HUMAN_V] + C.MODEL_ORDER):
                part = vot[vot.model.eq(model) & vot.raw_computable & np.isfinite(vot.raw_vot)]
                b = boxplot_stats(part.raw_vot.to_numpy(float), whis=(5, 95))[0]
                color = '#7A3E1D' if model == C.HUMAN_V else C.S.COLORS[model]
                for key in ['whislo', 'whishi']:
                    offscale(ax, b[key], yi, color, stem, model, key)
                row = vsum[vsum.Model.eq(model)].iloc[0]
                offscale(ax, row['Population Direct VOT-eq.'], yi + .13, color, stem, model, 'aggregate Direct VOT')
                offscale(ax, row['Median'], yi - .13, color, stem, model, 'Direct median')
        elif stem == 'Figure_10_gc_mnl_direct_elasticity':
            limits = [-6.182406517342555, .48445483379353854]
            for ax, models in zip(fig.axes, [C.MODEL_ORDER[:6], C.QWEN_ORDER]):
                ax.set_xlim(*limits)
                for yi, model in enumerate(models):
                    part = elasticity[elasticity.system.eq(model)]
                    b = boxplot_stats(part.elasticity.dropna().to_numpy(float), whis=(5, 95))[0]
                    for key in ['whislo', 'whishi']:
                        offscale(ax, b[key], yi, '#2F78A0', stem, model, key)
                    row = corr[corr.Model.eq(model)].iloc[0]; cfrow = cf[cf.Model.eq(model)].iloc[0]
                    offscale(ax, row['Direct Fare elasticity'], yi -.14, '#2F78A0', stem, model, 'Direct mean')
                    offscale(ax, row['GC-MNL Fare elasticity'], yi + .12, '#333333', stem, model, 'GC-MNL estimate', filled=False)
                    offscale(ax, cfrow['CF-MNL Fare elasticity'], yi + .24, '#E6A04B', stem, model, 'CF-MNL estimate')
        elif stem == 'Figure_11_gc_mnl_direct_vot':
            limits = [-50, 80]
            for ax, models in zip(fig.axes, [C.MODEL_ORDER[:6], C.QWEN_ORDER]):
                ax.set_xlim(*limits)
                for yi, model in enumerate(models):
                    part = vot[vot.model.eq(model) & vot.raw_computable & np.isfinite(vot.raw_vot)]
                    b = boxplot_stats(part.raw_vot.to_numpy(float), whis=(5, 95))[0]
                    for key in ['whislo', 'whishi']:
                        offscale(ax, b[key], yi, '#2F78A0', stem, model, key)
                    row = corr[corr.Model.eq(model)].iloc[0]; cfrow = cf[cf.Model.eq(model)].iloc[0]
                    offscale(ax, row['Direct VOT-eq.'], yi -.14, '#2F78A0', stem, model, 'aggregate Direct VOT')
                    offscale(ax, row['GC-MNL VOT'], yi + .12, '#333333', stem, model, 'GC-MNL estimate', filled=False)
                    offscale(ax, cfrow['CF-MNL VOT'], yi + .24, '#E6A04B', stem, model, 'CF-MNL estimate')
        return save(fig, stem, metadata={'xlim': limits, 'data_filtered_for_display': False,
            'offscale_marks': len([r for r in MANIFEST['offscale_display_marks'] if r['stem'] == stem])})
    C._save = intercept
    # Isolated process globals retain the original model keys; labels are changed at save.
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.4,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600})
    try:
        C.overall_vot_figure(vot, vsum, OUT, True)
        C.correspondence_elasticity_figure(corr, cf, elasticity, human, OUT, True)
        C.correspondence_vot_figure(corr, cf, vot, human, OUT, True)
    finally:
        C._save = old_save


def main():
    OUT.mkdir(exist_ok=True)
    input_paths = [A / n for n in ['direct_elasticity_cases.parquet', 'direct_elasticity_intervals.parquet',
                                    'direct_vot_cases.parquet', 'direct_vot_summary.csv', 'correspondence.csv',
                                    'cf_correspondence.csv', 'human_reference.json', 'response_curves.csv']]
    assert all(p.is_file() for p in input_paths), [str(p) for p in input_paths if not p.is_file()]
    MANIFEST['inputs_sha256'] = {str(p.relative_to(Q)): sha(p) for p in input_paths}
    MANIFEST['helper_sources'] = json.loads((OUT / 'source_snapshots/manifest.json').read_text())
    for item in MANIFEST['helper_sources']:
        assert sha(Q / item['snapshot']) == item['sha256']
    vot = pd.read_parquet(A / 'direct_vot_cases.parquet')
    elasticity = pd.read_parquet(A / 'direct_elasticity_cases.parquet')
    interval = pd.read_parquet(A / 'direct_elasticity_intervals.parquet')
    assert set(vot.model) == set([C.HUMAN_V] + C.MODEL_ORDER)
    assert vot.groupby('model').size().eq(3468).all()
    assert elasticity.groupby('system').size().eq(3468).all()
    stats = subplot_tables_and_stats(interval, vot)
    subgroup_figures(stats)
    response_figure()
    overall_and_correspondence(vot, elasticity)
    assert len(MANIFEST['figures']) == 6
    assert MANIFEST['inputs_sha256'] == {str(p.relative_to(Q)): sha(p) for p in input_paths}, 'Analysis inputs changed during figure generation'
    for item in MANIFEST['figures']:
        for folder in ['figures', 'figures_trimmed']:
            dest = PAPER / folder / (item['stem'] + '.pdf')
            assert dest.parent.is_dir()
            shutil.copy2(OUT / (item['stem'] + '.pdf'), dest)
            assert sha(dest) == item['pdf_sha256']
    MANIFEST.update(status='PASS', n_figures=6, n_subgroup_boxes=132, subgroup_quantile_checks=396,
                    cohort_cases=3468, cohort_respondents=651,
                    display_policy='Approved readable axis windows; every offscale displayed whisker or estimate is boundary-marked; no data trimming',
                    reference_role='Full retained-survey pooled reference; descriptive, not held-out Human predictions')
    (OUT / 'render_manifest.json').write_text(json.dumps(MANIFEST, indent=2), encoding='utf-8')
    print(json.dumps(MANIFEST, indent=2), flush=True)


if __name__ == '__main__':
    main()
