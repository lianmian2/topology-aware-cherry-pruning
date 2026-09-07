from __future__ import annotations

import argparse
from contextlib import ExitStack
import sys
import time

import pandas as pd
import torch
import numpy as np

import profile_pruning_components as runtime
from profile_pruning_structure import canonical
import prepare_pruning_decision_features as features

sys.path.insert(0, str(runtime.ROOT / '02_code/03_training'))
import train_pruning_segment_gnn_v2 as original


GROUP_KEYS = ['sample_id', 'tree_id', 'view', 'segment_id', 'candidate_type', 'is_cut_segment']


def main():
    parser = argparse.ArgumentParser(description='Saved fixed-split atomic GraphSAGE profiling; no training')
    parser.add_argument('--output', type=runtime.Path, default=runtime.ROOT / '04_results/pruning_decision/runtime_profiling_v1')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    report_root = runtime.baseline.ensure_dir(output / 'graphsage_variance_diagnostic') if args.diagnostic else output
    samples = runtime.read_json(output / 'sample_manifest.json')['samples']
    if runtime.read_json(output / 'structural_features/summary.json')['status'] != 'passed':
        raise RuntimeError('Structural features not complete')
    model_root = runtime.ROOT / '04_results/pruning_decision/pruning_decision_closeout_20260809/gnn_sage_branch_group_v1'
    frozen = runtime.read_json(model_root / 'summary.json')
    models, checkpoints, loading = [], [], []
    for seed in frozen['seeds']:
        path = model_root / 'checkpoints' / f'sage_full_seed{seed}.pt'
        runtime.sync()
        start = time.perf_counter()
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['config']
        if config['architecture'] != 'sage' or config['graph_context'] != 'branch_group':
            raise ValueError('Unexpected frozen architecture/context')
        model = original.SegmentGNN(len(checkpoint['feature_columns']), config['hidden'],
                                    config['dropout'], config['architecture'])
        model.load_state_dict(checkpoint['model_state'])
        model.to('cuda:0').eval()
        runtime.sync()
        loading.append({'seed': seed, 'seconds': time.perf_counter() - start,
                        'checkpoint': str(path.relative_to(runtime.ROOT)),
                        'sha256': runtime.baseline.sha256_file(path)})
        models.append(model)
        checkpoints.append(checkpoint)
    runtime.save_json(report_root / 'graphsage_loading.json', {
        'models': loading, 'scope': 'existing fixed-split atomic seed ensemble, not nested-CV or region models',
        'training_source_sha256': runtime.baseline.sha256_file(runtime.Path(original.__file__)),
        'wrapper_sha256': runtime.baseline.sha256_file(runtime.Path(__file__))})
    columns = checkpoints[0]['feature_columns']
    for checkpoint in checkpoints[1:]:
        if checkpoint['feature_columns'] != columns:
            raise ValueError('Feature schema mismatch between saved seeds')
    indices = [features.FEATURE_COLUMNS.index(column) for column in columns]
    stages = ('graphsage_input', 'graphsage_seed42', 'graphsage_ensemble')
    results = {stage: [] for stage in stages}
    audits = {stage: [] for stage in stages}
    controls = []
    for sample_index, sample in enumerate(samples):
        sample_id = sample['sample_id']
        rows, _, tensor = runtime.load_cache(output, sample_id, 'structural_features')
        metadata = pd.DataFrame(rows)
        disk_root = output / 'graphsage_input_assets'
        disk_path = disk_root / 'graphs' / sample_id / 'graph.pt'
        runtime.baseline.ensure_dir(disk_path.parent)
        torch.save(tensor, disk_path)
        disk_reference = original.load_split(disk_root, metadata, 'inference', indices, 'branch_group')

        def load_from_memory(*args, **kwargs):
            if runtime.Path(args[0]).resolve() != disk_path.resolve():
                raise ValueError('Unexpected file request inside in-memory graph interface')
            return tensor

        def build_input():
            with runtime.replace_callable(torch, 'load', load_from_memory):
                return original.load_split(disk_root, metadata, 'inference', indices, 'branch_group')

        graph, candidate_rows = build_input()
        disk_hash = runtime.fingerprint(canonical(disk_reference))
        if runtime.fingerprint(canonical((graph, candidate_rows))) != disk_hash:
            raise RuntimeError('Disk/in-memory GraphSAGE input mismatch')

        def predict_models(selected):
            frames = []
            for index in selected:
                checkpoint = checkpoints[index]
                normalized = dict(graph)
                normalized['x'] = (graph['x'] - checkpoint['mean']) / checkpoint['std']
                normalized = {key: value.to('cuda:0') for key, value in normalized.items()}
                frames.append(original.predict(models[index], normalized, candidate_rows))
            prediction = pd.concat(frames, ignore_index=True)
            if len(selected) > 1:
                prediction = prediction.groupby(GROUP_KEYS, as_index=False).score.mean()
            if not np.isfinite(prediction.score.to_numpy()).all():
                raise RuntimeError('Non-finite GraphSAGE score')
            return prediction[['sample_id', 'segment_id', 'score']].sort_values('score', ascending=False)

        single_index = [frozen['seeds'].index(42)]
        calls = {'graphsage_input': build_input,
                 'graphsage_seed42': lambda: predict_models(single_index),
                 'graphsage_ensemble': lambda: predict_models(range(len(models)))}
        for stage, call in calls.items():
            directory = runtime.baseline.ensure_dir(report_root / stage)
            if sample_index == 0:
                for _ in range(args.warmup):
                    call()
                    runtime.sync()
            reference = call()
            expected = runtime.fingerprint(canonical(reference))
            if args.diagnostic and stage != 'graphsage_input':
                for control_index in range(3):
                    control = call()
                    controls.append({'sample_id': sample_id, 'stage': stage, 'repeat': control_index + 1,
                                     'max_abs_score_delta': float(np.max(np.abs(reference.score.to_numpy() - control.score.to_numpy()))),
                                     'candidate_order_equal': reference.segment_id.tolist() == control.segment_id.tolist(),
                                     'exact_equal': expected == runtime.fingerprint(canonical(control))})
                runtime.save_json(report_root / 'uninstrumented_repeat_controls.json', controls)
            for repetition in range(args.repeats):
                selected = [] if stage == 'graphsage_input' else single_index if stage == 'graphsage_seed42' else list(range(len(models)))
                timers = [runtime.ForwardTimer(models[index]) for index in selected]
                with ExitStack() as stack:
                    for timer in timers:
                        stack.enter_context(timer.enabled())
                    runtime.sync()
                    started = time.perf_counter()
                    result = call()
                    runtime.sync()
                    seconds = time.perf_counter() - started
                observed = runtime.fingerprint(canonical(result))
                exact = expected == observed
                max_delta = 0.0
                order_equal = exact
                if stage != 'graphsage_input':
                    order_equal = reference.segment_id.tolist() == result.segment_id.tolist()
                    max_delta = float(np.max(np.abs(reference.score.to_numpy() - result.score.to_numpy())))
                equivalent = exact or (stage != 'graphsage_input' and order_equal and max_delta <= 1e-7)
                audits[stage].append({'sample_id': sample_id, 'repeat': repetition + 1,
                                      'reference_sha256': expected, 'profiled_sha256': observed,
                                      'exact_equal': exact, 'equivalent': equivalent,
                                      'candidate_order_equal': order_equal, 'max_abs_score_delta': max_delta,
                                      'disk_memory_input_equal': True})
                runtime.save_json(directory / 'equivalence_audit.json', audits[stage])
                if not equivalent:
                    runtime.save_json(directory / 'failure.json', audits[stage][-1])
                    if not args.diagnostic:
                        raise RuntimeError(f'GraphSAGE equality gate failed: {stage}/{sample_id}')
                forward = sum(stop - start for timer in timers for start, stop in timer.events)
                results[stage].append({'sample_id': sample_id, 'tree_id': sample['tree_id'],
                                       'repeat': repetition + 1, 'total_seconds': seconds,
                                       'model_forward_seconds': forward,
                                       'cpu_pre_post_and_transfers_seconds': seconds - forward})
                runtime.write_rows(directory / 'per_image_runtime.csv', results[stage])
                print(f'TIMED {stage} {sample_id} {seconds:.6f}s', flush=True)
            if stage != 'graphsage_input':
                ranked_dir = runtime.baseline.ensure_dir(report_root / 'ranked_candidates')
                reference.to_csv(ranked_dir / f'{sample_id}_{stage}.csv', index=False)
    for stage in stages:
        summary = runtime.summarize(results[stage])
        all_equal = all(row['equivalent'] for row in audits[stage])
        summary.update(status='passed' if all_equal else 'timed_but_score_equivalence_gate_failed',
                       exact_equivalence_all=all(row['exact_equal'] for row in audits[stage]),
                       candidate_order_preserved=all(row['candidate_order_equal'] for row in audits[stage]),
                       max_abs_score_delta=max(row['max_abs_score_delta'] for row in audits[stage]),
                       score_tolerance_atol=1e-7, score_tolerance_rtol=0.0)
        runtime.save_json(report_root / stage / 'summary.json', summary)
        print(stage, summary['total_seconds'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
