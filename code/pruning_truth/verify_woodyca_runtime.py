from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Independent standard-library audit of WoodyCA-Net runtime CSVs')
    parser.add_argument('--output', type=Path,
                        default=ROOT / '04_results/pruning_decision/runtime_profiling_woodyca_20260903')
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = read_json(output / 'run_manifest.json')
    frozen = read_json(output / 'sample_manifest.json')
    summary = read_json(output / 'summary.json')
    expected_ids = {sample['sample_id'] for sample in frozen['samples']}
    repeats = manifest['repeats_per_image_per_mode']
    table = []
    paths = {'end_to_end': output / 'end_to_end/per_image_runtime.csv'}
    paths.update({stage: output / 'components' / stage / 'per_image_runtime.csv'
                  for stage in summary['components']})
    for stage, path in paths.items():
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(expected_ids) * repeats, (stage, 'run count')
        assert {row['sample_id'] for row in rows} == expected_ids, (stage, 'image set')
        image_means = []
        for sample_id in sorted(expected_ids):
            group = [row for row in rows if row['sample_id'] == sample_id]
            assert sorted(int(row['repeat']) for row in group) == list(range(1, repeats + 1))
            values = [float(row['total_seconds']) for row in group]
            assert all(math.isfinite(value) and value > 0 for value in values)
            image_means.append(statistics.mean(values))
        calculated = {'mean': statistics.mean(image_means), 'sd': statistics.stdev(image_means),
                      'median': statistics.median(image_means), 'min': min(image_means), 'max': max(image_means)}
        recorded = summary['end_to_end'] if stage == 'end_to_end' else summary['components'][stage]
        for key, value in calculated.items():
            assert math.isclose(value, recorded['total_seconds'][key], rel_tol=1e-12, abs_tol=1e-12)
        table.append({'stage': stage, 'n_images': len(expected_ids), 'n_runs': len(rows), **calculated})
    audits = read_json(output / 'equivalence_audit.json')
    assert len(audits) == len(expected_ids) * repeats * 2
    assert len({(row['sample_id'], row['mode'], row['repeat']) for row in audits}) == len(audits)
    for row in audits:
        assert all(check['exact_equal'] for stage, check in row['stages'].items() if stage != 'graphsage_ensemble')
        assert row['stages']['graphsage_ensemble']['candidate_set_equal']
        assert row['stages']['graphsage_ensemble']['candidate_order_equal']
        for model in ('hgb_ranking', 'graphsage_ensemble'):
            path = output / 'ranked_candidates' / row['mode'] / f"{row['sample_id']}_{model}_repeat{row['repeat']}.csv"
            assert path.is_file()
    hashes = dict(manifest['source_sha256'])
    hashes.update(manifest['rulebook_sha256'])
    hashes.update({item['path']: item['sha256'] for item in manifest['weights'].values()})
    hashes.update({item['image_path']: item['image_sha256'] for item in frozen['samples']})
    assert all(sha256(ROOT / path) == expected for path, expected in hashes.items())
    ensure_dir(output)
    with (output / 'verified_runtime_table.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    result = {'status': 'verified', 'timing_statistics_recomputed': True,
              'n_component_stages': len(paths) - 1, 'n_images': len(expected_ids),
              'n_trees': len({sample['tree_id'] for sample in frozen['samples']}),
              'n_audited_calls': len(audits), 'n_verified_source_assets': len(hashes),
              'non_graphsage_exact': True, 'graphsage_order_exact': True,
              'graphsage_numerical_flag_retained': summary['status'],
              'end_to_end': table[0], 'verifier_sha256': sha256(Path(__file__))}
    (output / 'independent_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
