from __future__ import annotations

import importlib.util

import profile_pruning_components as runtime
from profile_pruning_structure import canonical
import prepare_pruning_decision_features as current


def main():
    output = runtime.ROOT / '04_results/pruning_decision/runtime_profiling_v1'
    snapshot = output / 'prepare_pruning_decision_features_before_wrapper.py'
    spec = importlib.util.spec_from_file_location('runtime_original_feature_source', snapshot)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    samples = runtime.read_json(output / 'sample_manifest.json')['samples']
    rows = []
    for sample in samples:
        path = runtime.DATASET / 'graphs' / sample['sample_id'] / 'decision_graph.json'
        before = original.process_graph(path, 'test')
        after = current.process_graph(path, 'test')
        left = runtime.fingerprint(canonical(before))
        right = runtime.fingerprint(canonical(after))
        rows.append({'sample_id': sample['sample_id'], 'original_sha256': left,
                     'wrapped_sha256': right, 'exact_equal': left == right})
        print(sample['sample_id'], left == right, flush=True)
    report = {'all_equal': all(row['exact_equal'] for row in rows), 'samples': rows,
              'original_source_sha256': runtime.baseline.sha256_file(snapshot),
              'current_source_sha256': runtime.baseline.sha256_file(runtime.Path(current.__file__)),
              'scope': 'all feature rows, full tensor payload and audit metadata from identical archived graphs; no raw data modified'}
    runtime.save_json(output / 'feature_interface_equivalence.json', report)
    return 0 if report['all_equal'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
