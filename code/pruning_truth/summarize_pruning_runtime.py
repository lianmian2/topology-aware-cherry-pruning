from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / '04_results/pruning_decision/runtime_profiling_v1'
STAGES = ('roi', 'woody', 'tape_support', 'bud', 'routing_geometry', 'bud_directions',
          'bud_flow_rerank', 'clip_refine', 'attachment', 'directed_graph', 'group_features',
          'atomic_structure', 'structural_features', 'region_features', 'hgb_ranking')


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    samples = read_json(OUTPUT / 'sample_manifest.json')['samples']
    expected_ids = {sample['sample_id'] for sample in samples}
    rows = []
    sources = [(stage, OUTPUT / stage) for stage in STAGES]
    diagnostic = OUTPUT / 'graphsage_variance_diagnostic'
    sources += [(stage, diagnostic / stage) for stage in ('graphsage_input', 'graphsage_seed42', 'graphsage_ensemble')]
    for stage, folder in sources:
        summary = read_json(folder / 'summary.json')
        with (folder / 'per_image_runtime.csv').open(encoding='utf-8-sig', newline='') as handle:
            records = list(csv.DictReader(handle))
        groups = {}
        for record in records:
            groups.setdefault(record['sample_id'], []).append(float(record['total_seconds']))
        if set(groups) != expected_ids or any(len(values) != 2 for values in groups.values()):
            raise ValueError(f'Incomplete or duplicated image repeats: {stage}')
        means = [statistics.mean(values) for values in groups.values()]
        for field, value in [('mean', statistics.mean(means)), ('sd', statistics.stdev(means)),
                             ('median', statistics.median(means))]:
            if not math.isclose(value, summary['total_seconds'][field], abs_tol=1e-10):
                raise ValueError(f'Statistical summary mismatch: {stage}/{field}')
        rows.append({'stage': stage, 'status': summary['status'], 'n_images': len(groups),
                     'n_trees': summary['n_trees'], 'n_timed_runs': len(records),
                     'mean_seconds': summary['total_seconds']['mean'],
                     'sd_seconds': summary['total_seconds']['sd'],
                     'median_seconds': summary['total_seconds']['median'],
                     'exact_equivalence_all': summary['exact_equivalence_all'],
                     'source': str(folder.relative_to(ROOT))})
    source_checks = {}
    for name in ('run_manifest.json', 'structural_run_manifest.json'):
        for relative, expected in read_json(OUTPUT / name)['source_sha256'].items():
            source_checks[relative] = sha256(ROOT / relative) == expected
    for sample in samples:
        source_checks[sample['image_path']] = sha256(ROOT / sample['image_path']) == sample['image_sha256']
    structural_manifest = read_json(OUTPUT / 'structural_run_manifest.json')
    thickness_rulebook = ROOT / '04_results/mask_topology_routing/gt_mask_thickness_rulebook_20260731/gt_mask_thickness_rulebook.json'
    source_checks[str(thickness_rulebook.relative_to(ROOT))] = sha256(thickness_rulebook) == structural_manifest['rulebook_sha256']
    binary_rulebook = Path(structural_manifest['binary_config']['rulebook_path'])
    source_checks[str(binary_rulebook.relative_to(ROOT))] = sha256(binary_rulebook) == structural_manifest['binary_config']['rulebook_sha256']
    for item in read_json(OUTPUT / 'run_manifest.json')['weights'].values():
        source_checks[item['path']] = sha256(ROOT / item['path']) == item['sha256']
    gnn_loading = read_json(diagnostic / 'graphsage_loading.json')
    for item in gnn_loading['models']:
        source_checks[item['checkpoint']] = sha256(ROOT / item['checkpoint']) == item['sha256']
    for relative, key in [('02_code/03_training/train_pruning_segment_gnn_v2.py', 'training_source_sha256'),
                          ('02_code/05_utils/pruning_truth/profile_pruning_graphsage.py', 'wrapper_sha256')]:
        source_checks[relative] = sha256(ROOT / relative) == gnn_loading[key]
    if not all(source_checks.values()):
        raise RuntimeError(f'Source changed: {[key for key, passed in source_checks.items() if not passed]}')
    controls = read_json(diagnostic / 'uninstrumented_repeat_controls.json')
    control_summary = {}
    for stage in ('graphsage_seed42', 'graphsage_ensemble'):
        selected = [row for row in controls if row['stage'] == stage]
        control_summary[stage] = {
            'n_uninstrumented_comparisons': len(selected),
            'max_abs_score_delta': max(row['max_abs_score_delta'] for row in selected),
            'candidate_order_preserved': all(row['candidate_order_equal'] for row in selected),
            'exact_matches': sum(row['exact_equal'] for row in selected)}
    e2e_dir = OUTPUT / 'atomic_e2e_same_process'
    e2e = None
    if (e2e_dir / 'summary.json').exists():
        e2e = read_json(e2e_dir / 'summary.json')
        with (e2e_dir / 'per_image_runtime.csv').open(encoding='utf-8-sig', newline='') as handle:
            records = list(csv.DictReader(handle))
        groups = {}
        for record in records:
            groups.setdefault(record['sample_id'], []).append(float(record['total_seconds']))
        if set(groups) != expected_ids or any(len(values) != 2 for values in groups.values()):
            raise ValueError('Incomplete E2E image repeats')
        means = [statistics.mean(values) for values in groups.values()]
        for field, value in [('mean', statistics.mean(means)), ('sd', statistics.stdev(means)),
                             ('median', statistics.median(means))]:
            if not math.isclose(value, e2e['total_seconds'][field], abs_tol=1e-10):
                raise ValueError(f'E2E statistical summary mismatch: {field}')
        manifest = read_json(e2e_dir / 'run_manifest.json')
        for relative, expected in manifest['source_sha256'].items():
            source_checks[relative] = sha256(ROOT / relative) == expected
        for item in manifest['weights'].values():
            source_checks[item['path']] = sha256(ROOT / item['path']) == item['sha256']
        if not all(source_checks.values()):
            raise RuntimeError('E2E source/weight hash mismatch')
        audits = read_json(e2e_dir / 'equivalence_audit.json')
        expected_runs = {(sample_id, repeat) for sample_id in expected_ids for repeat in (1, 2)}
        if len(audits) != len(expected_runs) or {(row['sample_id'], row['repeat']) for row in audits} != expected_runs:
            raise ValueError('Incomplete E2E equivalence audit')
        stage_checks = {}
        for stage in STAGES:
            if stage == 'region_features':
                continue
            stage_checks[stage] = all(row['stages'][stage]['exact_equal'] for row in audits)
        if not all(stage_checks.values()):
            raise ValueError(f'E2E stage-level equivalence failed: {stage_checks}')
        if not all(row['stages']['graphsage_ensemble']['candidate_order_equal'] for row in audits):
            raise ValueError('E2E GraphSAGE order changed')
        e2e['all_stage_exact_checks'] = stage_checks
    report = {'phase': 'component and continuous atomic E2E profiling' if e2e else 'component profiling', 'rows': rows, 'source_checks': source_checks,
              'uninstrumented_variance_controls': control_summary,
              'feature_interface_exact': read_json(OUTPUT / 'feature_interface_equivalence.json')['all_equal'],
              'end_to_end_measured': e2e is not None,
              'end_to_end': e2e,
              'end_to_end_source': str(e2e_dir.relative_to(ROOT)) if e2e else None,
              'region_scope': 'excluded from continuous pipeline by author; no region weight required',
              'pending': ([] if e2e else ['continuous atomic end-to-end profiling']),
              'retained_audit_flags': ['GraphSAGE strict score-equivalence gate; exact candidate order and numerical score differences reported separately',
                                       'separate-process perception caches can differ; same-process uninstrumented reference used for timer equivalence'],
              'scope': 'deployed legacy segmentation and saved fixed-split atomic rankers; not paper-final region or channel-only runtime'}
    ensure_dir(OUTPUT)
    with (OUTPUT / 'component_runtime_summary.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / 'component_runtime_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'rows': rows, 'controls': control_summary, 'sources_unchanged': True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
